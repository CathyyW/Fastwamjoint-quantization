from types import SimpleNamespace
import pytest
import torch
from fastwam_steerquant import policy as policy_module
from fastwam_steerquant.state import DenoiseCallState


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fail_once = False

    def _predict_joint_noise(self, value):
        if self.fail_once:
            self.fail_once = False
            raise RuntimeError("interrupted chunk")
        return value + self._wam_call_tracker.state.require()

    def infer_action(self, *, num_inference_steps, **kwargs):
        value = torch.zeros(1)
        for _ in range(num_inference_steps):
            value = self._predict_joint_noise(value)
        return {"action": value}


def make_policy(monkeypatch):
    model = FakeModel()
    state = DenoiseCallState(2)
    metadata = {"model_config": {"sigma_shift": None, "model": {"action_horizon": 4, "num_video_frames": 5}}}
    monkeypatch.setattr(policy_module, "load_deployment", lambda *a, **kw: (model, state, metadata))
    adapter = SimpleNamespace(build_model=lambda *a, **kw: model,
        infer_kwargs=lambda model, record, seed: {"num_inference_steps": record.get("steps", 2),
              "action_horizon": record.get("horizon", 4), "num_video_frames": 5})
    return policy_module.QuantizedPolicy("fixture", adapter, device="cpu")


def test_policy_resets_call_state_between_chunks_and_after_error(monkeypatch):
    policy = make_policy(monkeypatch)
    for _ in range(3):
        torch.testing.assert_close(policy.infer({})["action"], torch.ones(1))
        assert policy.state.index is None
    policy.model.fail_once = True
    with pytest.raises(RuntimeError, match="interrupted"):
        policy.infer({})
    assert policy.state.index is None
    torch.testing.assert_close(policy.infer({})["action"], torch.ones(1))


def test_policy_rejects_mismatched_geometry_and_reentrant_calls(monkeypatch):
    policy = make_policy(monkeypatch)
    with pytest.raises(ValueError, match="num_inference_steps"):
        policy.infer({"steps": 3})
    with pytest.raises(ValueError, match="action_horizon"):
        policy.infer({"horizon": 8})
    policy._lock.acquire()
    try:
        with pytest.raises(RuntimeError, match="serialized"):
            policy.infer({})
    finally:
        policy._lock.release()
