"""Integration into an existing real-robot loop; no hardware actions here."""
import threading
import torch
from .deployment import load_deployment
from .state import FastWAMCallTracker
from .rollout_graph import LiveJointDiTGraph


class QuantizedPolicy:
    def __init__(self, deployment, adapter, *, device="cuda", construct_device="meta", cuda_graph=False):
        self.model, self.state, self.metadata = load_deployment(
            deployment, adapter.build_model, device=device, construct_device=construct_device)
        if callable(getattr(adapter, "finalize_model", None)):
            adapter.finalize_model(self.model, device=device)
        self.adapter = adapter
        self.tracker = FastWAMCallTracker(self.model, self.state)
        self.tracker.install()
        self.model._wam_call_tracker = self.tracker
        if cuda_graph:
            LiveJointDiTGraph(self.model, num_calls=self.state.num_calls).install()
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
