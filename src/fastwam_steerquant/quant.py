from __future__ import annotations

import torch


def symmetric_fake_quant(
    values: torch.Tensor,
    *,
    bits: int,
    scale: torch.Tensor,
    ste: bool = False,
) -> torch.Tensor:
    if not 2 <= int(bits) <= 8:
        raise ValueError("bits must be in [2, 8].")
    qmax = 2 ** (int(bits) - 1) - 1
    normalized = values / scale.clamp_min(1e-8)
    if ste:
        rounded = normalized + (normalized.round() - normalized).detach()
    else:
        rounded = normalized.round()
    return rounded.clamp(-qmax, qmax) * scale


def fake_quant_activation_per_tensor(
    values: torch.Tensor,
    *,
    bits: int,
    clipping_range: torch.Tensor | float | None = None,
    ste: bool = False,
) -> torch.Tensor:
    fp32 = values.float()
    clip = fp32.detach().abs().amax() if clipping_range is None else torch.as_tensor(
        clipping_range, device=fp32.device, dtype=torch.float32
    )
    if clip.numel() != 1 or not torch.isfinite(clip) or clip <= 0:
        raise ValueError("clipping_range must be one finite positive scalar.")
    qmax = 2 ** (int(bits) - 1) - 1
    return symmetric_fake_quant(
        fp32, bits=bits, scale=clip.reshape(()) / qmax, ste=ste
    ).to(values.dtype)


def fake_quant_weight_per_channel(
    weight: torch.Tensor,
    *,
    bits: int,
    ste: bool = False,
) -> torch.Tensor:
    fp32 = weight.float()
    qmax = 2 ** (int(bits) - 1) - 1
    scales = fp32.detach().abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / qmax
    return symmetric_fake_quant(fp32, bits=bits, scale=scales, ste=ste).to(weight.dtype)


def quantize_weight_per_channel(
    weight: torch.Tensor,
    *,
    bits: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    fp32 = weight.detach().float()
    qmax = 2 ** (int(bits) - 1) - 1
    scales = fp32.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / qmax
    qweight = (fp32 / scales).round().clamp(-qmax, qmax).to(torch.int8)
    return qweight, scales
