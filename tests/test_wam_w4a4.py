from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from fastwam_steerquant.kernels.w4a4 import pack_signed_int4
from fastwam_steerquant.kernels.w4a4 import wam_ops as ops


CASES = [(32, 1024, 3072), (32, 3072, 1024), (32, 1024, 4096),
         (32, 4096, 1024), (129, 1024, 3072), (129, 3072, 3072),
         (294, 3072, 3072), (294, 3072, 14336), (294, 14336, 3072)]
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def inputs(dtype, m, k, n, batch=1):
    torch.manual_seed(m + k + n)
    x = torch.randn(batch, m, k, device="cuda", dtype=dtype) * 0.5
    qw = torch.randint(-7, 8, (n, k), device="cuda", dtype=torch.int8)
    ws = torch.full((n,), 1 / 512, device="cuda")
    inverse = torch.pow(2., torch.randint(-1, 2, (k,), device="cuda").float())
    bias = torch.randn(n, device="cuda", dtype=dtype) * 0.01
    counts = (16, 16) if m == 32 else (128, 1) if m == 129 else (98, 98, 98)
    gains = torch.ones(10, len(counts), device="cuda")
    gains[0] = torch.tensor([0.5, 2., 1.][:len(counts)], device="cuda")
    gains[9] = torch.tensor([2., 0.5, 1.][:len(counts)], device="cuda")
    scales = torch.linspace(0.125, 0.25, 10, device="cuda")
    return x, qw, ws, bias, inverse, scales, gains, counts


def oracle(x, qw, ws, bias, inverse, scales, gains, counts, call):
    gain = torch.repeat_interleave(gains[call], torch.tensor(counts, device=x.device))[None, :, None]
    qa = (x.float() * inverse * (gain / scales[call])).round().clamp(-7, 7)
    acc = F.linear(qa.double(), qw.double()).float()
    return acc * (scales[call] / gain) * ws + bias.float()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("m,k,n", CASES)
def test_wam_all_production_shapes_schedule_graph_and_tiles(monkeypatch, dtype, m, k, n):
    args = inputs(dtype, m, k, n, batch=2 if m == 129 else 1)
    x, qw, ws, bias, inv, scales, gains, counts = args
    packed = pack_signed_int4(qw)
    for call in (0, 9):
        expected = oracle(*args, call).to(dtype)
        results = []
        for tile in ("64", "128", "auto"):
            monkeypatch.setenv("FASTWAM_W4A4_TILE", tile)
            actual = ops.symmetric_scheduled_stream_linear(x, packed, ws, bias, inv, scales, gains, counts, call)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = ops.symmetric_scheduled_stream_linear(x, packed, ws, bias, inv, scales, gains, counts, call)
            graph.replay()
            torch.testing.assert_close(captured, actual, rtol=0, atol=0)
            results.append(actual)
        torch.testing.assert_close(results[0], results[1], rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("m,k,n", [(32, 1024, 4096), (294, 3072, 14336)])
def test_adaln_fusion(dtype, m, k, n):
    x, qw, ws, bias, inv, scales, gains, counts = inputs(dtype, m, k, n)
    # A strided modulation view matches FastWAM's unbound AdaLN tensor.
    modulation = torch.randn(len(counts), 6, k, dtype=dtype, device="cuda") * 0.1
    scale, shift = modulation[:, 0], modulation[:, 1]
    actual = ops.symmetric_scheduled_stream_adaln_linear(
        x, pack_signed_int4(qw), ws, bias, inv, scales, gains, counts, 9, scale, shift, 1e-6)
    xf = x.float()
    normalized = (xf - xf.mean(-1, keepdim=True)) * torch.rsqrt(xf.var(-1, unbiased=False, keepdim=True) + 1e-6)
    transformed = normalized * (1 + scale.float().repeat_interleave(counts[0], 0)) + shift.float().repeat_interleave(counts[0], 0)
    expected = oracle(transformed, qw, ws, bias, inv, scales, gains, counts, 9).to(dtype)
    relative_rms = (actual.float() - expected.float()).square().mean().sqrt() / expected.float().square().mean().sqrt()
    assert relative_rms < 0.003  # FP32 LayerNorm reduction and A4 boundary ties.
    packed = pack_signed_int4(qw)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = ops.symmetric_scheduled_stream_adaln_linear(
            x, packed, ws, bias, inv, scales, gains, counts, 9, scale, shift, 1e-6)
    graph.replay()
    torch.testing.assert_close(captured, actual, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("m,k,n", [(32, 4096, 1024), (294, 14336, 3072)])
def test_gate_residual_fusion(dtype, m, k, n):
    args = inputs(dtype, m, k, n)
    x, qw, ws, bias, inv, scales, gains, counts = args
    residual = torch.randn(1, m, n, device="cuda", dtype=dtype)
    gate = (torch.randn(len(counts), 6, n, device="cuda", dtype=dtype) * 0.1)[:, 0]
    assert not gate.is_contiguous()
    actual = ops.symmetric_scheduled_stream_gate_residual_linear(
        x, pack_signed_int4(qw), ws, bias, inv, scales, gains, counts, 9, residual, gate)
    expected = (oracle(*args, 9) * gate.float().repeat_interleave(counts[0], 0) + residual.float()).to(dtype)
    torch.testing.assert_close(actual, expected, rtol=0.008, atol=0.001)
    packed = pack_signed_int4(qw)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = ops.symmetric_scheduled_stream_gate_residual_linear(
            x, packed, ws, bias, inv, scales, gains, counts, 9, residual, gate)
    graph.replay()
    torch.testing.assert_close(captured, actual, rtol=0, atol=0)


def test_invalid_layout_and_call_are_rejected():
    x, qw, ws, bias, inv, scales, gains, counts = inputs(torch.bfloat16, 129, 1024, 3072)
    packed = pack_signed_int4(qw)
    with pytest.raises(ValueError, match="layout"):
        ops.symmetric_scheduled_stream_linear(x, packed, ws, bias, inv, scales, gains, (127, 1), 0)
    with pytest.raises(RuntimeError, match="index"):
        ops.symmetric_scheduled_stream_linear(x, packed, ws, bias, inv, scales, gains, counts, 10)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_runtime_preserves_checkpoint_d_gamma_and_a4_clipping(monkeypatch, dtype):
    from fastwam_steerquant.checkpoint import QuantizedSite
    from fastwam_steerquant.runtime import WAMQuantLinear
    from fastwam_steerquant.state import DenoiseCallState
    from fastwam_steerquant.streams import FastWAMStreamConfig
    # Default BF16 cuBLAS may reduce partial sums in BF16 (observed ~0.1%
    # relative RMS). Disable that only for this exact arithmetic check;
    # native INT32 accumulation and deployment benchmark settings are unchanged.
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_bf16_reduced_precision_reduction", False)
    source = torch.nn.Linear(256, 8, bias=False, device="cuda", dtype=dtype)
    entry = QuantizedSite(
        0, "linear", "action", "self_attn.o", ("action_chunk_0", "action_chunk_1"),
        torch.randint(-7, 8, (8, 256), dtype=torch.int8), torch.full((8, 1), 1 / 64),
        torch.ones(256) * 2, torch.tensor([[0.5, 2.], [2., 0.5]]),
        torch.tensor([1.75, 3.5]), torch.full((2, 2), 0.5), 0., 0., (0., 0.), (0., 0.))
    state = DenoiseCallState(2)
    kwargs = dict(state=state, stream_config=FastWAMStreamConfig(), weight_bits=4, activation_bits=4)
    native = WAMQuantLinear(source, entry, backend="cutlass_w4a4", **kwargs)
    reference = WAMQuantLinear(source, entry, backend="reference", **kwargs)
    assert native.weight.dtype == torch.uint8 and native.weight.shape == (8, 128)
    assert native.in_features == 256
    torch.testing.assert_close(native.clipping_ranges.cpu(), entry.clipping_ranges / 7)
    x = torch.randn(2, 32, 256, device="cuda", dtype=dtype)
    with pytest.raises(RuntimeError, match="not been set"):
        native(x)
    for call in (0, 1):
        state.set(call)
        torch.testing.assert_close(native(x), reference(x), rtol=0, atol=0)
    with pytest.raises(ValueError, match="does not match"):
        WAMQuantLinear(source, entry, backend="cutlass_w4a8", **kwargs)
