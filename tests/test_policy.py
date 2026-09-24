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


def test_policy_fusion_installed_before_graph_and_rht_rejected(monkeypatch):
    model = FakeModel()
    metadata = {"sites": [{"rotation": "none", "operation": name}
                          for name in ("ffn.0", "ffn.2", "self_attn.o")]}
    monkeypatch.setattr(policy_module, 'load_deployment',
                        lambda *a, **kw: (model, DenoiseCallState(2), metadata))
    events = []
    adapter = SimpleNamespace(build_model=lambda *a, **kw: model,
                              enable_block_fusion_dispatch=lambda m: events.append('dispatch'))
    def enable(m, flag):
        assert flag
        events.append('fusion')
        return 3
    monkeypatch.setattr(policy_module, 'set_wam_block_fusions', enable)
    monkeypatch.setattr(policy_module, 'LiveJointDiTGraph',
                        lambda *a, **kw: SimpleNamespace(install=lambda: events.append('graph')))
    policy = policy_module.QuantizedPolicy('fixture', adapter, fuse_block=True, cuda_graph=True)
    assert policy.fused_modules == 3 and events == ['dispatch', 'fusion', 'graph']
    events.clear()
    metadata['sites'][0]['rotation'] = 'rht'
    with pytest.raises(ValueError, match='RHT block fusion'):
        policy_module.QuantizedPolicy('fixture', adapter, fuse_block=True)
    assert not events


def test_policy_consumes_profile_before_loading_and_installs_graph_without_fusion(monkeypatch):
    from pathlib import Path
    from fastwam_steerquant import runtime_profile as rp
    from fastwam_steerquant.kernels.w4a8 import loader
    import os
    for name in ('FASTWAM_RUNTIME_PROFILE', 'FASTWAM_W4A8_TILE', 'FASTWAM_W4A8_EXPERIMENTAL_SMALL_TILE'):
        monkeypatch.setenv(name, '')
        monkeypatch.delenv(name)
    monkeypatch.setattr(rp, '_configured_tile', None)
    monkeypatch.setattr(loader, 'load_extension', SimpleNamespace(cache_info=lambda: SimpleNamespace(currsize=0)))
    events = []
    model = FakeModel()
    metadata = dict(weight_bits=4, activation_bits=8, sites=[dict(rotation='none')])
    def load(*args, **kw):
        assert os.environ['FASTWAM_W4A8_TILE'] == '64'
        assert kw['construct_device'] == 'cpu'
        events.append('load')
        return model, DenoiseCallState(2), metadata
    monkeypatch.setattr(policy_module, 'load_deployment', load)
    monkeypatch.setattr(policy_module, 'set_wam_block_fusions', lambda m, v: events.append(('fusion', v)))
    monkeypatch.setattr(policy_module, 'LiveJointDiTGraph',
                        lambda *a, **kw: SimpleNamespace(install=lambda: events.append('graph')))
    adapter = SimpleNamespace(build_model=lambda *a, **kw: model)
    policy = policy_module.QuantizedPolicy('fixture', adapter,
        runtime_profile=Path(__file__).resolve().parents[1]/'configs/real_robot_w4a8_runtime.json')
    assert events == ['load', ('fusion', False), 'graph']
    assert policy.runtime_settings['cuda_graph'] is True
    assert policy.runtime_settings['fuse_block'] is False and policy.fused_modules == 0
