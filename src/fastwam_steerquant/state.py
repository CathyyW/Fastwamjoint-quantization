from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn


@dataclass
class DenoiseCallState:
    num_calls: int
    index: int | None = None

    def __post_init__(self) -> None:
        if self.num_calls <= 0:
            raise ValueError("num_calls must be positive.")

    def set(self, index: int) -> None:
        index = int(index)
        if not 0 <= index < self.num_calls:
            raise IndexError(f"Denoise call {index} is outside [0, {self.num_calls}).")
        self.index = index

    def require(self) -> int:
        if self.index is None:
            raise RuntimeError("Denoise call state has not been set.")
        return self.index

    def clear(self) -> None:
        self.index = None


class FastWAMCallTracker:
    """Set call state around FastWAMJoint._predict_joint_noise invocations."""

    def __init__(self, model: nn.Module, state: DenoiseCallState) -> None:
        self.model = model
        self.state = state
        self.total_calls = 0
        self._original: Any | None = None

    def install(self) -> None:
        if self._original is not None:
            raise RuntimeError("Call tracker is already installed.")
        original = getattr(self.model, "_predict_joint_noise")
        self._original = original

        def tracked(*args: Any, **kwargs: Any):
            self.state.set(self.total_calls % self.state.num_calls)
            self.total_calls += 1
            return original(*args, **kwargs)

        setattr(self.model, "_predict_joint_noise", tracked)

    def reset(self) -> None:
        self.total_calls = 0
        self.state.clear()

    def remove(self) -> None:
        if self._original is not None:
            setattr(self.model, "_predict_joint_noise", self._original)
            self._original = None
        self.state.clear()

    def __enter__(self) -> "FastWAMCallTracker":
        self.install()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.remove()


def differentiable_joint_denoise(
    model: nn.Module,
    *,
    latents_video: torch.Tensor,
    latents_action: torch.Tensor,
    first_frame_latents: torch.Tensor,
    context: torch.Tensor,
    context_mask: torch.Tensor,
    num_inference_steps: int,
    state: DenoiseCallState,
    sigma_shift: float | None = None,
) -> torch.Tensor:
    """Differentiable counterpart of FastWAMJoint's joint denoising loop.

    VAE/text preparation stays outside this function. The returned action is
    not detached, so final-action VJPs can reach any selected Linear output.
    """

    if state.num_calls != int(num_inference_steps):
        raise ValueError("state.num_calls must equal num_inference_steps.")
    video_steps, video_deltas = model.infer_video_scheduler.build_inference_schedule(
        num_inference_steps=num_inference_steps,
        device=latents_video.device,
        dtype=latents_video.dtype,
        shift_override=sigma_shift,
    )
    action_steps, action_deltas = model.infer_action_scheduler.build_inference_schedule(
        num_inference_steps=num_inference_steps,
        device=latents_action.device,
        dtype=latents_action.dtype,
        shift_override=sigma_shift,
    )
    fuse = bool(getattr(model.video_expert, "fuse_vae_embedding_in_latents", False))
    for call, (tv, dv, ta, da) in enumerate(zip(video_steps, video_deltas, action_steps, action_deltas)):
        state.set(call)
        timestep_video = tv.unsqueeze(0).to(device=latents_video.device, dtype=latents_video.dtype)
        timestep_action = ta.unsqueeze(0).to(device=latents_action.device, dtype=latents_action.dtype)
        latent_t, latent_h, latent_w = latents_video.shape[-3:]
        patch_t, patch_h, patch_w = (int(value) for value in model.video_expert.patch_size)
        tokens_per_frame = (latent_h // patch_h) * (latent_w // patch_w)
        mask = model._build_mot_attention_mask(
            video_seq_len=(latent_t // patch_t) * tokens_per_frame,
            action_seq_len=latents_action.shape[1],
            video_tokens_per_frame=tokens_per_frame,
            device=latents_video.device,
        )
        pred_video, pred_action = model._joint_denoise_core(
            latents_video=latents_video,
            latents_action=latents_action,
            timestep_video=timestep_video,
            timestep_action=timestep_action,
            context=context,
            context_mask=context_mask,
            attention_mask=mask,
            fuse_vae_embedding_in_latents=fuse,
            action_condition=None,
        )
        latents_video = model.infer_video_scheduler.step(pred_video, dv, latents_video)
        latents_action = model.infer_action_scheduler.step(pred_action, da, latents_action)
        latents_video = torch.cat(
            [first_frame_latents, latents_video[:, :, first_frame_latents.shape[2] :]],
            dim=2,
        )
    state.clear()
    return latents_action
