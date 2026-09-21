from __future__ import annotations

import torch

from .loader import load_extension


def pack_signed_int4(qweight: torch.Tensor) -> torch.Tensor:
    if qweight.ndim != 2 or qweight.dtype != torch.int8 or qweight.shape[-1] % 2:
        raise ValueError("W4A8 expects INT8 [N, even K] weights containing signed INT4 values.")
    if not qweight.is_cuda and qweight.numel() and (qweight.min() < -8 or qweight.max() > 7):
        raise ValueError("Quantized weights must lie in [-8, 7].")
    nibble = qweight.to(torch.uint8) & 0x0F
    return (nibble[:, ::2] | (nibble[:, 1::2] << 4)).contiguous()


def _common(x, packed, scales, bias, inverse):
    if x.dtype not in (torch.bfloat16, torch.float16) or not x.is_cuda or x.ndim < 2:
        raise ValueError("W4A8 expects CUDA BF16/FP16 inputs with token and channel axes.")
    k, n = x.shape[-1], packed.shape[0]
    if packed.dtype != torch.uint8 or packed.shape != (n, k // 2) or k % 32 or n % 8:
        raise ValueError(f"Unsupported FastWAM W4A8 shape: K={k}, N={n}.")
    if packed.device != x.device or scales.device != x.device or scales.dtype != torch.float32 or scales.shape != (n,):
        raise ValueError("Packed weights and float32 per-channel scales must be on the input device.")
    if inverse.dtype != torch.float32 or inverse.device != x.device or inverse.shape != (k,):
        raise ValueError("Precomputed inverse D must be CUDA float32 [K].")
    if bias is not None and (bias.dtype != x.dtype or bias.device != x.device or bias.shape != (n,)):
        raise ValueError("The native bias must match the input precision and output width.")
    return x.contiguous()


def symmetric_dynamic_linear(x, packed, scales, bias, inverse):
    x = _common(x, packed, scales, bias, inverse)
    return load_extension().symmetric_dynamic(x, packed, scales, bias, inverse, True)


def symmetric_scheduled_stream_linear(x, packed, scales, bias, inverse, activation_scales, gains, token_counts, call):
    x = _common(x, packed, scales, bias, inverse)
    if len(token_counts) != gains.shape[-1] or sum(token_counts) != x.shape[-2]:
        raise ValueError("WAM token layout does not match the input and gain schedule.")
    if activation_scales.device != x.device or activation_scales.dtype != torch.float32 or activation_scales.ndim != 1:
        raise ValueError("WAM clipping scale schedule must be CUDA float32 [calls].")
    if gains.device != x.device or gains.dtype != torch.float32 or gains.shape[0] != activation_scales.numel():
        raise ValueError("WAM gains must be CUDA float32 [calls, streams].")
    if len(set(token_counts)) == 1:
        span = int(token_counts[0])
    elif len(token_counts) == 2 and token_counts[0] > 0 and token_counts[1] > 0:
        # Negative span selects FastWAM's unequal text/proprio producer.
        span = -int(token_counts[1])
    else:
        raise ValueError("Native WAM supports equal video/action streams or text/proprio streams.")
    return load_extension().symmetric_static_stream_scheduled(
        x, packed, scales, bias, inverse,
        activation_scales.contiguous(), gains.contiguous(), span, int(call), True,
    )


def _equal_stream_span(x, gains, token_counts):
    if len(token_counts) != gains.shape[-1] or not token_counts or sum(token_counts) != x.shape[-2]:
        raise ValueError("WAM fused stream layout does not cover the input token axis.")
    if len(set(token_counts)) != 1:
        raise ValueError("The Cosmos AdaLN/gate W4A8 producer requires equal-length FastWAM streams.")
    return int(token_counts[0])


def symmetric_scheduled_stream_adaln_linear(
    x, packed, scales, bias, inverse, activation_scales, gains, token_counts, call,
    adaln_scale, adaln_shift, epsilon,
):
    """Move affine-free LayerNorm and FastWAM's MLP AdaLN into A8 producer."""
    x = _common(x, packed, scales, bias, inverse)
    span = _equal_stream_span(x, gains, token_counts)
    if (adaln_scale.shape != adaln_shift.shape or adaln_scale.shape[-1] != x.shape[-1]
            or adaln_scale.dtype != x.dtype or adaln_scale.device != x.device):
        raise ValueError("AdaLN scale and shift must match the input dtype and hidden width.")
    rows = x.numel() // x.shape[-1]
    modulation_rows = adaln_scale.numel() // x.shape[-1]
    if modulation_rows <= 0 or rows % modulation_rows:
        raise ValueError("AdaLN groups must cover complete input rows.")
    return load_extension().symmetric_static_stream_scheduled_adaln(
        x, adaln_scale, adaln_shift, rows // modulation_rows, float(epsilon),
        packed, scales, bias, inverse, activation_scales, gains, span, int(call), True,
    )


def symmetric_scheduled_stream_gate_residual_linear(
    x, packed, scales, bias, inverse, activation_scales, gains, token_counts, call,
    residual, gate,
):
    """Move FastWAM's x+gate*projection into the W4A8 GEMM epilogue."""
    x = _common(x, packed, scales, bias, inverse)
    span = _equal_stream_span(x, gains, token_counts)
    if (residual.shape != (*x.shape[:-1], packed.shape[0]) or residual.dtype != x.dtype
            or residual.device != x.device):
        raise ValueError("Residual must match the projection output shape and dtype.")
    if gate.dtype != x.dtype or gate.device != x.device or gate.shape[-1] != packed.shape[0]:
        raise ValueError("Gate must match the projection output dtype and width.")
    rows = x.numel() // x.shape[-1]
    gate_rows = gate.numel() // packed.shape[0]
    if gate_rows <= 0 or rows % gate_rows:
        raise ValueError("Gate groups must cover complete output rows.")
    return load_extension().symmetric_static_stream_scheduled_gate_residual(
        x, packed, scales, bias, inverse, activation_scales, gains, span, int(call), True,
        residual.contiguous(), gate, rows // gate_rows,
    )
