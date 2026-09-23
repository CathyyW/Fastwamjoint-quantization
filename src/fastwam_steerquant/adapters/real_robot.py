"""Offline adapter for the audited pack/stack robot source handoff.

No ROS, robot SDK, RPC or network operations. External model code is hash-pinned
and imported from the handoff, never silently from another installed qi package.
"""
from __future__ import annotations

import copy
import hashlib
import importlib
import json
from pathlib import Path
import sys

import torch

from .base import validate_action_scale
from .fastwam import prepare_from_infer_kwargs
from .robot_core import DenseJointCoreMixin


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_file(path, expected):
    if not expected or sha256_file(path) != expected:
        raise ValueError(f"SHA256 mismatch or missing expected hash: {path}")


def activate_handoff(root, manifest_sha256):
    root = Path(root).expanduser().resolve()
    manifest = root / "SHA256SUMS"
    verify_file(manifest, manifest_sha256)
    listed = set()
    for line in manifest.read_text().splitlines():
        checksum, name = line.split(maxsplit=1)
        path = (root / name.lstrip("*")).resolve()
        if not path.is_relative_to(root) or path in listed:
            raise ValueError("Unsafe or duplicate handoff manifest path.")
        verify_file(path, checksum)
        listed.add(path)
    source = root / "src"
    # Also reject unmanifested Python that could shadow the verified closure.
    if any(p.resolve() not in listed for p in source.rglob("*.py")):
        raise ValueError("Unmanifested Python source in handoff.")
    for name, module in tuple(sys.modules.items()):
        if name == "qi" or name.startswith("qi.") or name == "offline_preprocessing":
            location = getattr(module, "__file__", None)
            locations = [location] if location else list(getattr(module, "__path__", []))
            if not locations or any(not Path(p).resolve().is_relative_to(source) for p in locations):
                raise RuntimeError(f"Another {name} is already imported; use a fresh process.")
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    importlib.invalidate_caches()
    return importlib.import_module("offline_preprocessing")


def build_structure(architecture, *, device="cpu"):
    """CPU structure only. Never invokes a pretrained/checkpoint loader.

    Each large component is cast separately, limiting CPU FP32 temporaries.
    CPU construction is intentional: upstream schedulers and plain RoPE/VAE
    attributes are not meta-safe. Packed replacement happens before GPU transfer.
    """
    if str(device) != "cpu":
        raise ValueError("Real robot builder requires construct_device='cpu', not meta/CUDA.")
    from qi.models.wan22.fastwam_joint import FastWAMJoint
    from qi.models.wan22.wan_video_dit import WanVideoDiT
    from qi.models.wan22.action_dit import ActionDiT
    from qi.models.wan22.mot import MoT
    from qi.models.wan22.wan_video_vae import WanVideoVAE38

    class SteerQuantRealJoint(DenseJointCoreMixin, FastWAMJoint):
        pass

    arch = copy.deepcopy(architecture)
    if arch["video_dit_config"].get("action_conditioned", False):
        raise ValueError("Only unconditioned FastWAMJoint video expert is supported.")
    with torch.device("cpu"):
        video = WanVideoDiT(**arch["video_dit_config"]).to(dtype=torch.bfloat16)
        action = ActionDiT(**arch["action_dit_config"]).to(dtype=torch.bfloat16)
        mot = MoT({"video": video, "action": action},
                  mot_checkpoint_mixed_attn=arch["mot_checkpoint_mixed_attn"])
        vae = WanVideoVAE38().to(dtype=torch.bfloat16)
        scheduler_args = {}
        for expert in ("video", "action"):
            for key, value in arch[f"{expert}_scheduler"].items():
                scheduler_args[f"{expert}_{key}"] = value
        model = SteerQuantRealJoint(
            video_expert=video, action_expert=action, mot=mot, vae=vae,
            text_encoder=None, tokenizer=None,
            text_dim=arch["video_dit_config"]["text_dim"],
            proprio_dim=arch["proprio_dim"], device="cpu", torch_dtype=torch.bfloat16,
            loss_lambda_video=arch["loss"]["lambda_video"],
            loss_lambda_action=arch["loss"]["lambda_action"], **scheduler_args)
    return model.eval().requires_grad_(False)


class RealRobotAdapter:
    def __init__(self, config):
        self.config = copy.deepcopy(config)
        self.task = config["real_robot"]
        self.preprocessing = activate_handoff(config["handoff_root"], self.task["manifest_sha256"])
        for key in ("dataset_stats", "context_cache", "training_config"):
            verify_file(config["assets"][key], self.task["asset_sha256"][key])
        if (config["num_calls"] != 10 or config.get("sigma_shift") is not None
                or config["model"] != {"dtype": "bfloat16", "action_horizon": 32, "num_video_frames": 9}):
            raise ValueError("This audited adapter requires BF16, 10 steps, 32 actions, 9 video frames, shift=None.")
        from hydra.utils import instantiate
        from qi.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
        self.processor = instantiate(self.task["processor"])
        self.processor.set_normalizer_from_stats(
            load_dataset_stats_from_json(config["assets"]["dataset_stats"]))
        self.processor.eval()
        self.context, self.context_mask = self.preprocessing.load_cached_context(
            Path(config["assets"]["context_cache"]).parent,
            self.preprocessing.format_prompt(self.task["prompt"]), 128)
        if self.context.shape != (128, 4096) or not torch.isfinite(self.context).all():
            raise ValueError("Expected finite [128,4096] cached context.")

    def calibration_readiness(self):
        confirmed = self.config.get("calibration_approval", {})
        required = ("stats_checkpoint_binding", "training_split", "image_pipeline", "vae_identity")
        return [key for key in required if not isinstance(confirmed.get(key), str)
                or not confirmed[key].strip()]

    def require_calibration_ready(self):
        missing = self.calibration_readiness()
        if missing:
            raise ValueError("Calibration needs reviewed evidence: " + ", ".join(missing))
        self.action_scale()

    def action_scale(self):
        spec = self.config.get("normalized_action_scale", {})
        if not spec.get("evidence"):
            raise ValueError("Provide training-normalized action std and its evidence; raw std is not action_scale.")
        value = validate_action_scale(spec.get("values", []), device="cpu")
        if value.shape != (14,):
            raise ValueError("Expected 14 normalized action standard deviations.")
        return value

    def build_model(self, model_config, *, device="cpu"):
        # Ignore old source checkpoint paths embedded during export. The current
        # machine's handoff supplies code/assets, but must have identical identity.
        self.validate_deployment_config(model_config)
        return build_structure(model_config["real_robot"]["architecture"], device=device)

    def validate_deployment_config(self, model_config):
        for key in ("num_calls", "sigma_shift", "model", "expected_sites", "stream_config"):
            if model_config[key] != self.config[key]:
                raise ValueError(f"Deployment and adapter disagree on {key}.")
        for key in ("manifest_sha256", "asset_sha256", "architecture", "processor", "prompt", "task",
                    "source_checkpoint_sha256", "vae_checkpoint_sha256"):
            if model_config["real_robot"][key] != self.task[key]:
                raise ValueError(f"Deployment and adapter disagree on {key}.")

    def load_model(self, *, device="cuda"):
        self.require_calibration_ready()
        assets = self.config["assets"]
        verify_file(assets["source_checkpoint"], self.task["source_checkpoint_sha256"])
        verify_file(assets["vae_checkpoint"], self.task["vae_checkpoint_sha256"])
        model = self.build_model(self.config, device="cpu")
        payload = torch.load(assets["source_checkpoint"], map_location="cpu", weights_only=True, mmap=True)
        if set(payload) != {"mot", "proprio_encoder", "step", "torch_dtype"}:
            raise ValueError("Unexpected source checkpoint schema; audit before loading.")
        model.mot.load_state_dict(payload["mot"], strict=True)
        model.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
        del payload
        from qi.models.wan22.helpers.state_dict_converters import wan_video_vae_state_dict_converter
        vae = torch.load(assets["vae_checkpoint"], map_location="cpu", weights_only=True, mmap=True)
        model.vae.load_state_dict(wan_video_vae_state_dict_converter(vae), strict=True)
        del vae
        model.to(device=device).eval().requires_grad_(False)
        self.finalize_model(model, device=device)
        return model

    def finalize_model(self, model, *, device):
        model.device = torch.device(device)
        model.torch_dtype = torch.bfloat16
        model._sync_vae_scale_device()
        model.video_expert.freqs = tuple(x.to(device=device) for x in model.video_expert.freqs)
        model.action_expert.freqs = model.action_expert.freqs.to(device=device)

    def infer_kwargs(self, model, record, *, seed):
        import numpy as np
        if record.get("task") != self.task["task"] or record.get("prompt") != self.task["prompt"]:
            raise ValueError("Observation task/prompt differs from adapter.")
        channel_order = record.get("image_channel_order", "RGB")
        if channel_order not in ("RGB", "as_stored"):
            raise ValueError("Convert known BGR to RGB in the dataset reader first.")
        if channel_order == "as_stored" and self.config.get("dataset", {}).get("channel_policy") != "as_stored":
            raise ValueError("Stored-channel observations require explicit as_stored experiment approval.")
        images = {}
        for name in ("cam_high", "cam_left_wrist", "cam_right_wrist"):
            array = np.asarray(record["images"][name])
            if array.dtype != np.uint8 or array.shape != (480, 640, 3):
                raise ValueError(f"{name} must be uint8 [480,640,3] with the declared channel convention.")
            images[name] = array
        state = np.asarray(record["state"], dtype=np.float32)
        if state.shape != (14,) or not np.isfinite(state).all():
            raise ValueError("State must contain 14 finite physical-coordinate values.")
        obs = self.preprocessing.WAMObservation(**images, state=state, prompt=record["prompt"])
        image = self.preprocessing.preprocess_real_images(obs, model.device, model.torch_dtype)
        proprio = self.preprocessing.normalize_proprio(self.processor, state).unsqueeze(0)
        return dict(prompt=None, input_image=image, proprio=proprio,
                    context=self.context, context_mask=self.context_mask,
                    action_horizon=32, num_video_frames=9, num_inference_steps=10,
                    sigma_shift=None, text_cfg_scale=1.0, seed=seed, rand_device="cpu",
                    tiled=False, expert_cache=False)

    def prepare_inputs(self, model, record, *, seed):
        return prepare_from_infer_kwargs(model, self.infer_kwargs(model, record, seed=seed), seed=seed)

    def denormalize_actions(self, action):
        # Production processor's statistics are CPU tensors. Do not postprocess
        # or truncate the 32 actions; the existing robot client owns execution.
        return self.preprocessing.denormalize_actions(self.processor, action.detach().cpu())

    def dataset_records(self):
        from .robot_dataset import records_from_manifest
        return records_from_manifest(self.config)


def make_adapter(config):
    return RealRobotAdapter(config)
