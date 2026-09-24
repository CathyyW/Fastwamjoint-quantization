"""Optional contract tests against the real source snapshot, using tiny experts.

FASTWAM_ROBOT_HANDOFF=/path/to/fastwam_calibration_handoff enables these tests.
No real checkpoint, CUDA, VAE forward, dataset or robot is required.
"""
import copy
import json
import os
from pathlib import Path

import pytest
import torch
from torch import nn

from fastwam_steerquant.adapters.real_robot import (
    RealRobotAdapter, activate_handoff, build_structure, sha256_file)
from fastwam_steerquant.state import DenoiseCallState, differentiable_joint_denoise


@pytest.fixture(scope="module")
def handoff():
    path = os.environ.get("FASTWAM_ROBOT_HANDOFF")
    if not path:
        pytest.skip("Set FASTWAM_ROBOT_HANDOFF to test the audited external model source")
    root = Path(path)
    activate_handoff(root, sha256_file(root / "SHA256SUMS"))
    return root


@pytest.fixture
def tiny_model(handoff, monkeypatch):
    from qi.models.wan22 import wan_video_vae
    from qi.models.wan22.fastwam_joint import FastWAMJoint
    from omegaconf import OmegaConf

    class TinyVAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Linear(1, 1)
            self.mean, self.std = torch.zeros(1), torch.ones(1)

    def no_load(*args, **kwargs):
        raise AssertionError("Structure builder attempted to load weights")

    monkeypatch.setattr(wan_video_vae, "WanVideoVAE38", TinyVAE)
    monkeypatch.setattr(torch, "load", no_load)
    monkeypatch.setattr(FastWAMJoint, "from_wan22_pretrained", no_load)
    source = OmegaConf.to_container(OmegaConf.load(
        handoff / "configs/train_config_joint_pack3_infer.yaml"), resolve=False)["model"]
    keys = ("video_dit_config", "action_dit_config", "mot_checkpoint_mixed_attn",
            "proprio_dim", "video_scheduler", "action_scheduler", "loss")
    arch = {k: copy.deepcopy(source[k]) for k in keys}
    for name in ("video_dit_config", "action_dit_config"):
        arch[name].update(hidden_dim=32, ffn_dim=64, num_heads=2, attn_head_dim=12,
                          num_layers=1, text_dim=16, freq_dim=8, use_gradient_checkpointing=False)
    arch["mot_checkpoint_mixed_attn"] = False
    model = build_structure(arch, device="cpu")
    assert model.video_expert is model.mot.mixtures["video"]
    assert model.dit is model.mot
    return model


def inputs():
    return dict(latents_video=torch.randn(1, 48, 3, 4, 4, dtype=torch.bfloat16),
                latents_action=torch.randn(1, 4, 14, dtype=torch.bfloat16),
                timestep_video=torch.tensor([500.], dtype=torch.bfloat16),
                timestep_action=torch.tensor([500.], dtype=torch.bfloat16),
                context=torch.randn(1, 5, 16, dtype=torch.bfloat16),
                context_mask=torch.ones(1, 5, dtype=torch.bool),
                fuse_vae_embedding_in_latents=True)


def test_dense_core_matches_unmodified_upstream_and_supports_vjp(tiny_model):
    from qi.models.wan22.fastwam import FastWAM
    data = inputs()
    reference = FastWAM._predict_joint_noise(tiny_model, **data)
    actual = tiny_model._predict_joint_noise(**data)
    for left, right in zip(reference, actual):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    data["latents_action"].requires_grad_(True)
    data["latents_video"].requires_grad_(True)
    mask = tiny_model._build_mot_attention_mask(12, 4, 4, torch.device("cpu"))
    out = tiny_model._joint_denoise_core(**data, attention_mask=mask)
    grads = torch.autograd.grad(out[1].float().square().mean(),
                                (data["latents_action"], data["latents_video"]))
    assert all(torch.isfinite(g).all() and torch.count_nonzero(g) for g in grads)


def test_opt_in_post_block_dispatch_matches_audited_dense_handoff(tiny_model):
    from fastwam_steerquant.adapters.robot_fusion import install_robot_block_fusion_dispatch
    data = inputs()
    with torch.inference_mode():
        expected = tiny_model._predict_joint_noise(**data)
        install_robot_block_fusion_dispatch(tiny_model)
        actual = tiny_model._predict_joint_noise(**data)
    for left, right in zip(actual, expected):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_three_step_differentiable_rollout_matches_upstream(tiny_model):
    from qi.models.wan22.fastwam import FastWAM
    data = inputs()
    video, action = data["latents_video"], data["latents_action"]
    first = video[:, :, :1].clone()
    tv, dv = tiny_model.infer_video_scheduler.build_inference_schedule(
        num_inference_steps=3, device="cpu", dtype=torch.bfloat16, shift_override=None)
    ta, da = tiny_model.infer_action_scheduler.build_inference_schedule(
        num_inference_steps=3, device="cpu", dtype=torch.bfloat16, shift_override=None)
    with torch.inference_mode():
        for v, vd, a, ad in zip(tv, dv, ta, da):
            pv, pa = FastWAM._predict_joint_noise(tiny_model, latents_video=video,
                latents_action=action, timestep_video=v.unsqueeze(0), timestep_action=a.unsqueeze(0),
                context=data["context"], context_mask=data["context_mask"], fuse_vae_embedding_in_latents=True)
            video = tiny_model.infer_video_scheduler.step(pv, vd, video)
            video[:, :, :1] = first
            action = tiny_model.infer_action_scheduler.step(pa, ad, action)
    seed_action = data["latents_action"].clone().requires_grad_(True)
    actual = differentiable_joint_denoise(tiny_model,
        latents_video=data["latents_video"], latents_action=seed_action,
        first_frame_latents=first, context=data["context"], context_mask=data["context_mask"],
        num_inference_steps=3, state=DenoiseCallState(3))
    torch.testing.assert_close(actual, action, rtol=0, atol=0)
    assert torch.isfinite(torch.autograd.grad(actual.float().sum(), seed_action)[0]).all()


@pytest.mark.parametrize("cache", [True, False])
def test_dense_core_rejects_unsupported_routes(tiny_model, cache):
    with pytest.raises(ValueError):
        tiny_model._predict_joint_noise(**inputs(),
            expert_cache_state=object() if cache else None,
            gt_action=None if cache else torch.ones(1))


def test_builder_rejects_meta_without_imports():
    with pytest.raises(ValueError, match="construct_device"):
        build_structure({}, device="meta")


def test_readiness_and_action_scale_do_not_guess():
    adapter = RealRobotAdapter.__new__(RealRobotAdapter)
    adapter.config = {}
    with pytest.raises(ValueError, match="reviewed evidence"):
        adapter.require_calibration_ready()
    adapter.config["calibration_approval"] = {k: "reviewed source" for k in adapter.calibration_readiness()}
    with pytest.raises(ValueError, match="training-normalized"):
        adapter.require_calibration_ready()
    adapter.config["normalized_action_scale"] = {"values": [1.] * 14, "evidence": "test fixture only"}
    adapter.require_calibration_ready()


@pytest.mark.parametrize("task", ["pack", "stack"])
def test_actual_preprocessing_stats_and_cached_text(handoff, task):
    import numpy as np
    from omegaconf import OmegaConf
    task_data = json.loads((handoff / "tasks.json").read_text())["tasks"][task]
    source = OmegaConf.to_container(OmegaConf.load(handoff / task_data["architecture_config"]), resolve=False)
    assets = {"dataset_stats": str(handoff / task_data["dataset_stats"]),
              "context_cache": str(handoff / task_data["context_cache_file"]),
              "training_config": str(handoff / task_data["architecture_config"])}
    config = {"handoff_root": str(handoff), "assets": assets, "num_calls": 10, "sigma_shift": None,
        "model": {"dtype": "bfloat16", "action_horizon": 32, "num_video_frames": 9},
        "real_robot": {"manifest_sha256": sha256_file(handoff / "SHA256SUMS"),
                       "asset_sha256": {k: sha256_file(v) for k, v in assets.items()},
                       "processor": source["data"]["train"]["processor"],
                       "task": task, "prompt": task_data["prompt"]}}
    adapter = RealRobotAdapter(config)
    model = type("Model", (), {"device": torch.device("cpu"), "torch_dtype": torch.bfloat16})()
    images = {name: np.full((480, 640, 3), value, dtype=np.uint8)
              for name, value in (("cam_high", 255), ("cam_left_wrist", 0), ("cam_right_wrist", 128))}
    state = np.zeros(14, dtype=np.float32)
    record = {"task": task, "prompt": task_data["prompt"], "images": images, "state": state}
    kwargs = adapter.infer_kwargs(model, record, seed=42)
    assert kwargs["input_image"].shape == (1, 3, 384, 320)
    assert (kwargs["input_image"][:, :, :256] == 1).all()
    assert (kwargs["input_image"][:, :, 256:, :160] == -1).all()
    expected_state = adapter.preprocessing.normalize_proprio(adapter.processor, state).unsqueeze(0)
    torch.testing.assert_close(kwargs["proprio"], expected_state, rtol=0, atol=0)
    cache = torch.load(assets["context_cache"], weights_only=True)
    assert kwargs["context_mask"].all()
    assert (kwargs["context"][~cache["mask"].bool()] == 0).all()
    action = torch.zeros(32, 14)
    assert adapter.denormalize_actions(action).shape == (32, 14)
    record["task"] = "wrong"
    with pytest.raises(ValueError, match="task/prompt"):
        adapter.infer_kwargs(model, record, seed=42)
