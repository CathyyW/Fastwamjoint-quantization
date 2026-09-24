#!/usr/bin/env python3
"""Small shape smoke: fused WAM AdaLN/gate against the unfused BF16 path."""

from __future__ import annotations

import torch
import pytest

from fastwam_steerquant.checkpoint import QuantizedSite
from fastwam_steerquant.runtime import WAMQuantLinear
from fastwam_steerquant.state import DenoiseCallState
from fastwam_steerquant.streams import FastWAMStreamConfig


def make_layer(operation, k, n, state):
    source = torch.nn.Linear(k, n, dtype=torch.bfloat16, device="cuda")
    entry = QuantizedSite(
        site_index=0, module_name=f"video_expert.blocks.0.{operation}",
        expert="video", operation=operation,
        stream_names=("latent_frame_0", "latent_frame_1", "latent_frame_2"),
        qweight=torch.randint(-7, 8, (n, k), dtype=torch.int8),
        weight_scales=torch.full((n, 1), 0.0004),
        input_scale=torch.ones(k),
        gamma_gains=torch.ones(10, 3),
        clipping_ranges=torch.ones(10),
        token_fractions=torch.full((10, 3), 1 / 3),
        d_initial_loss=0.0, d_final_loss=0.0,
        gamma_initial_losses=(0.0,) * 10, gamma_final_losses=(0.0,) * 10,
    )
    return WAMQuantLinear(
        source, entry, state=state, stream_config=FastWAMStreamConfig(),
        weight_bits=4, activation_bits=8, backend="cutlass_w4a8", fuse_block=True,
    ).eval()


def compare(actual, expected, label):
    diff = (actual.float() - expected.float()).abs()
    rms = diff.square().mean().sqrt() / expected.float().square().mean().sqrt().clamp_min(1e-6)
    print(label, "max_abs", float(diff.max()), "relative_rms", float(rms), flush=True)
    # Cosmos FP32 producer differs from BF16 intermediate rounds in FastWAM.
    # Oracle uses the exact unfused WAM INT GEMM, not an FP full-precision GEMM.
    if not torch.isfinite(actual).all() or float(rms) > 0.065:
        raise AssertionError(f"Fused {label} drifted materially from unfused WAM.")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_native_fusions_and_graph_match_historical_tolerance():
    torch.manual_seed(42)
    state = DenoiseCallState(10)
    state.set(0)
    x = torch.randn(1, 6, 1024, device="cuda", dtype=torch.bfloat16) * 0.1
    scale = torch.randn(1, 1, 1024, device="cuda", dtype=torch.bfloat16) * 0.03
    shift = torch.randn_like(scale) * 0.03
    ffn0 = make_layer("ffn.0", 1024, 4096, state)
    with torch.inference_mode():
        norm = torch.nn.functional.layer_norm(x, (1024,), eps=1e-6)
        unfused = ffn0(norm * (1 + scale) + shift)
        fused = ffn0.forward_adaln(x, scale, shift, 1e-6)
        compare(fused, unfused, "MLP LayerNorm/AdaLN")

        proj_input = torch.randn(1, 6, 4096, device="cuda", dtype=torch.bfloat16) * 0.1
        residual = torch.randn(1, 6, 1024, device="cuda", dtype=torch.bfloat16) * 0.1
        gate = torch.randn(1, 1, 1024, device="cuda", dtype=torch.bfloat16) * 0.1
        ffn2 = make_layer("ffn.2", 4096, 1024, state)
        unfused = residual + gate * ffn2(proj_input)
        fused = ffn2.forward_gate_residual(proj_input, residual, gate)
        compare(fused, unfused, "MLP gate/residual")
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            ffn0.forward_adaln(x, scale, shift, 1e-6)
            ffn2.forward_gate_residual(proj_input, residual, gate)
        graph.replay()
        torch.cuda.synchronize()
        print("two native fusion entries also capture/replay in CUDA Graph", flush=True)
