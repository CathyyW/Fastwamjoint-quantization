import pytest
import torch

from fastwam_steerquant.rollout_graph import LiveJointDiTGraph
from fastwam_steerquant.state import DenoiseCallState, FastWAMCallTracker


def test_validation_strict_default_and_explicit_warning(caplog):
    reference = (torch.ones(4),)
    actual = (torch.ones(4) + 0.1,)
    strict = LiveJointDiTGraph(None)
    with pytest.raises(AssertionError):
        strict._validate_outputs(actual, reference, 0)
    warn = LiveJointDiTGraph(None, allow_numerical_mismatch=True)
    warn._validate_outputs(actual, reference, 0)
    assert len(warn.validation_records) == 1
    assert not warn.validation_records[0]["passed"]
    assert warn.validation_records[0]["rmse"] == pytest.approx(0.1)
    assert "ACCEPTED by explicit opt-in" in caplog.text


@pytest.mark.parametrize("allow", [False, True])
@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_validation_rejects_nonfinite_in_both_modes(allow, bad):
    graph = LiveJointDiTGraph(None, allow_numerical_mismatch=allow)
    for actual, expected in [((torch.tensor([bad]),), (torch.ones(1),)),
                             ((torch.ones(1),), (torch.tensor([bad]),))]:
        with pytest.raises(RuntimeError, match="Nonfinite"):
            graph._validate_outputs(actual, expected, 0)


def test_warning_mode_still_rejects_output_layout_changes():
    graph = LiveJointDiTGraph(None, allow_numerical_mismatch=True)
    with pytest.raises(RuntimeError, match="layout"):
        graph._validate_outputs((torch.ones(2),), (torch.ones(3),), 0)
    with pytest.raises(RuntimeError, match="count"):
        graph._validate_outputs((torch.ones(2),), (), 0)


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.state = DenoiseCallState(2)
        self._wam_call_tracker = FastWAMCallTracker(self, self.state)
        self._wam_call_tracker.install()

    def _joint_denoise_core(self, x, timestep, context, mask, flag=True):
        y = x * (self.state.require() + 1) + timestep + context * mask
        return y, y + 1

    def _predict_joint_noise(self, **kwargs):
        return self._joint_denoise_core(**kwargs)

    @torch.no_grad()
    def infer_action(self, x, timestep, context, mask, flag=True, num_inference_steps=2,
                     text_cfg_scale=1.0, compile_action_infer=False):
        self._wam_call_tracker.reset()
        for _ in range(num_inference_steps):
            x, _ = self._predict_joint_noise(x=x, timestep=timestep, context=context, mask=mask, flag=flag)
        return x


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_live_graph_updates_all_inputs_and_call_state_and_preserves_outputs():
    model = Tiny().cuda().eval()
    inputs = [dict(x=torch.full((4,), float(i), device="cuda"),
                   timestep=torch.tensor(float(i), device="cuda"),
                   context=torch.arange(4, device="cuda").float() + i,
                   mask=torch.tensor([i % 2, 1, 0, 1], device="cuda")) for i in range(3)]
    expected = [model.infer_action(**values) for values in inputs]
    original = model.infer_action
    graph = LiveJointDiTGraph(model, num_calls=2).install()
    actual = [model.infer_action(**values) for values in inputs]
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert len(graph.entries) == 2 and graph.replays == 6
    assert model._wam_call_tracker.total_calls == 2
    with pytest.raises(ValueError, match="layout/constants"):
        model.infer_action(**dict(inputs[0], flag=False))
    with pytest.raises(ValueError, match="layout/constants"):
        model.infer_action(**dict(inputs[0], x=torch.zeros(8, device="cuda")))
    with pytest.raises(ValueError, match="schedule changed"):
        model.infer_action(**dict(inputs[0], num_inference_steps=3))
    with pytest.raises(ValueError, match="text_cfg_scale"):
        model.infer_action(**dict(inputs[0], text_cfg_scale=2))
    graph.remove()
    assert model.infer_action == original
