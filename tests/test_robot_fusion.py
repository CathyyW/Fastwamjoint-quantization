import pytest
import torch
from torch import nn

from fastwam_steerquant.adapters.robot_fusion import install_robot_block_fusion_dispatch
from fastwam_steerquant.runtime import set_wam_block_fusions


class Projection(nn.Linear):
    def __init__(self):
        super().__init__(8, 8)
        self.supports_fused_adaln = False
        self.supports_fused_gate_residual = False
        self.calls = []

    def forward_adaln(self, x, scale, shift, epsilon):
        self.calls.append('adaln')
        return self(nn.functional.layer_norm(x, (8,), eps=epsilon) * (1 + scale) + shift)

    def forward_gate_residual(self, x, residual, gate):
        self.calls.append('gate')
        return residual + gate * self(x)


class Cross(nn.Module):
    def forward(self, x, context, ctx_mask):
        self.mask = ctx_mask
        return x * .1 + context.mean(1, keepdim=True)


def fixture():
    block = nn.Module()
    block.self_attn = nn.Module()
    block.self_attn.o = Projection()
    block.norm2 = nn.LayerNorm(8, elementwise_affine=False)
    block.norm3 = nn.LayerNorm(8, elementwise_affine=False)
    block.cross_attn = Cross()
    block.ffn = nn.Sequential(Projection(), nn.GELU(), Projection())
    block.gate = lambda x, gate, y: x + gate * y
    block.eval()
    model = nn.Module()
    model.mot = nn.Module()
    model.mot._apply_expert_post_block = lambda *a, **kw: None
    model.eval()
    return model, block


@pytest.mark.parametrize('context', [False, True])
def test_robot_dispatch_calls_three_fusions_with_unchanged_semantics(context):
    torch.manual_seed(7)
    model, block = fixture()
    x, attn, gate, shift, scale, mlp_gate = [torch.randn(1, 6, 8) for _ in range(6)]
    payload = {'context': torch.randn(1, 4, 8), 'mask': torch.ones(1, 6, 4)} if context else None
    install_robot_block_fusion_dispatch(model)
    args = (block, x, attn, gate, shift, scale, mlp_gate, payload)
    expected = model.mot._apply_expert_post_block(*args)
    block.self_attn.o.supports_fused_gate_residual = True
    block.ffn[0].supports_fused_adaln = True
    block.ffn[2].supports_fused_gate_residual = True
    actual = model.mot._apply_expert_post_block(*args, version1_layer_index=0, version1_expert_name='video')
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert block.self_attn.o.calls == ['gate']
    assert block.ffn[0].calls == ['adaln']
    assert block.ffn[2].calls == ['gate']
    if context:
        assert block.cross_attn.mask.shape == (1, 1, 6, 4)
    assert not hasattr(type(model.mot), '_steerquant_block_fusion_dispatch')


def test_fusion_rejects_sparse_training_and_graph_mutation():
    model, block = fixture()
    model._rollout_graph = object()
    with pytest.raises(RuntimeError, match='before CUDA Graph'):
        install_robot_block_fusion_dispatch(model)
    with pytest.raises(RuntimeError, match='after CUDA Graph'):
        set_wam_block_fusions(model, True)
    del model._rollout_graph
    model.mot.version1_sparse_controller = object()
    with pytest.raises(ValueError, match='dense'):
        install_robot_block_fusion_dispatch(model)
    model.mot.version1_sparse_controller = None
    model.train()
    with pytest.raises(RuntimeError, match='inference-only'):
        install_robot_block_fusion_dispatch(model)
