"""Integration into an existing real-robot loop; no hardware actions here."""
import threading
import logging
import os
import torch
from .deployment import load_deployment
from .state import FastWAMCallTracker
from .rollout_graph import LiveJointDiTGraph
from .runtime import set_wam_block_fusions
from .runtime_profile import resolve_runtime_profile, validate_profile_metadata


class QuantizedPolicy:
    def __init__(self, deployment, adapter, *, device="cuda", construct_device=None,
                 cuda_graph=None, fuse_block=None, runtime_profile=None):
        options, self.runtime_profile = resolve_runtime_profile(
            profile=runtime_profile, construct_device=construct_device,
            cuda_graph=cuda_graph, fuse_block=fuse_block)
        construct_device, cuda_graph, fuse_block = (
            options['construct_device'], options['cuda_graph'], options['fuse_block'])
        self.model, self.state, self.metadata = load_deployment(
            deployment, adapter.build_model, device=device, construct_device=construct_device)
        validate_profile_metadata(self.runtime_profile, self.metadata)
        if callable(getattr(adapter, "finalize_model", None)):
            adapter.finalize_model(self.model, device=device)
        self.adapter = adapter
        self.fused_modules = 0
        if self.runtime_profile is not None:
            # Explicitly disable even if a custom adapter set flags in finalize_model.
            set_wam_block_fusions(self.model, False)
        if fuse_block:
            dispatch = getattr(adapter, "enable_block_fusion_dispatch", None)
            if not callable(dispatch):
                raise ValueError("Adapter must explicitly support block fusion dispatch.")
            if any(s["rotation"] != "none" for s in self.metadata["sites"]):
                raise ValueError("RHT block fusion is not validated; leave fuse_block=False.")
            dispatch(self.model)
            self.fused_modules = set_wam_block_fusions(self.model, True)
            expected = sum(s["operation"] in ("self_attn.o", "ffn.0", "ffn.2")
                           for s in self.metadata["sites"])
            if not expected or self.fused_modules != expected:
                raise RuntimeError("Block fusion site count does not match the deployment.")
        self.tracker = FastWAMCallTracker(self.model, self.state)
        self.tracker.install()
        self.model._wam_call_tracker = self.tracker
        if cuda_graph:
            LiveJointDiTGraph(self.model, num_calls=self.state.num_calls).install()
        self.runtime_settings = {**options, "w4a8_tile": os.environ.get('FASTWAM_W4A8_TILE', 'auto'),
                                 "fused_modules": self.fused_modules,
                                 "profile": self.runtime_profile['name'] if self.runtime_profile else None}
        logging.warning("[SteerQuant runtime] %s", self.runtime_settings)
        self._lock = threading.Lock()

    @torch.inference_mode()
    def infer(self, record, *, seed=42):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Use one policy instance per serialized inference loop.")
        try:
            self.tracker.reset()
            kwargs = self.adapter.infer_kwargs(self.model, record, seed=seed)
            if kwargs.get("num_inference_steps") != self.state.num_calls:
                raise ValueError("infer_kwargs must explicitly match calibrated num_inference_steps.")
            if float(kwargs.get("text_cfg_scale", 1)) != 1:
                raise ValueError("Current calibration/deployment protocol requires text_cfg_scale=1.")
            if kwargs.get("sigma_shift") != self.metadata["model_config"].get("sigma_shift"):
                raise ValueError("Inference sigma schedule differs from calibration configuration.")
            for key in ("action_horizon", "num_video_frames"):
                configured = self.metadata["model_config"].get("model", {}).get(key)
                if configured is not None and kwargs.get(key) != configured:
                    raise ValueError(f"Inference {key} differs from calibration configuration.")
            result = self.model.infer_action(**kwargs)
            if self.tracker.total_calls != self.state.num_calls:
                raise RuntimeError("Actual denoising calls differ from calibration.")
            return result
        finally:
            self.state.clear()
            self._lock.release()
