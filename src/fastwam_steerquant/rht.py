"""Randomized Hadamard transforms for local WAM calibration and inference.

The 'rht' mode uses had12 x H256 / had28 x H512 at widths 3072 / 14336.
The 'block' utility is retained only for mathematical regression comparisons.
Constants' original source and license are recorded in third_party/README.md.
"""

from __future__ import annotations

import hashlib
import math
from functools import lru_cache

import torch

from .rht_constants import HAD12_MASKS, HAD28_MASKS


@lru_cache(maxsize=2)
def official_hadamard_matrix(order: int) -> torch.Tensor:
    """Exact matrices reconstructed from the pinned Hadamard masks (CPU FP32)."""
    masks = {12: HAD12_MASKS, 28: HAD28_MASKS}[order]
    return torch.tensor([[1.0 if mask & (1 << j) else -1.0
                          for j in range(order)] for mask in masks])


def _official_hadamard(values: torch.Tensor, *, transpose: bool = False) -> torch.Tensor:
    width = values.shape[-1]
    if width > 0 and width & (width - 1) == 0:
        return _normalized_block_hadamard(values)
    if width not in (3072, 14336):
        raise ValueError(f"Official FastWAM RHT supports power-of-two widths and 3072/14336, got {width}.")
    order = 12 if width == 3072 else 28
    blocks = values.reshape(-1, order, width // order)
    transformed = _normalized_block_hadamard(blocks)
    matrix = official_hadamard_matrix(order).to(device=values.device, dtype=values.dtype)
    if transpose:
        matrix = matrix.T
    return (matrix @ transformed).reshape_as(values) / math.sqrt(order)


class _RandomizedOfficialHadamard(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, signs):
        ctx.save_for_backward(signs)
        return _official_hadamard(values * signs.to(values))

    @staticmethod
    def backward(ctx, grad_output):
        (signs,) = ctx.saved_tensors
        # had12/had28 need not be symmetric: use their transposes here.
        return _official_hadamard(grad_output, transpose=True) * signs.to(grad_output), None


def randomized_official_hadamard_transform(values: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    if values.ndim == 0 or signs.shape != (values.shape[-1],):
        raise ValueError("Official RHT requires one sign per input channel.")
    return _RandomizedOfficialHadamard.apply(values, signs)


def rht_transform(values: torch.Tensor, signs: torch.Tensor, rotation: str) -> torch.Tensor:
    if rotation == "rht":
        return randomized_official_hadamard_transform(values, signs)
    if rotation == "block":
        return randomized_block_hadamard_transform(values, signs)
    raise ValueError(f"Unknown RHT rotation: {rotation}")


def rht_block_size(width: int) -> int:
    """Return the largest power-of-two block that exactly divides ``width``."""

    width = int(width)
    if width <= 0:
        raise ValueError(f"RHT input width must be positive, got {width}.")
    block_size = width & -width
    if block_size < 2:
        raise ValueError(f"RHT requires an even input width, got {width}.")
    return block_size


def rht_signs(
    in_features: int,
    *,
    seed: int,
    module_name: str,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Create reproducible per-site Rademacher signs without Python hashes."""

    rht_block_size(in_features)
    digest = hashlib.sha256(f"{int(seed)}:{module_name}".encode()).digest()
    local_seed = int.from_bytes(digest[:8], "little") % (2**63 - 1)
    generator = torch.Generator(device="cpu").manual_seed(local_seed)
    signs = torch.randint(0, 2, (int(in_features),), generator=generator, dtype=torch.int8)
    signs = signs.mul_(2).sub_(1).float()
    return signs.to(device=device) if device is not None else signs


def _normalized_block_hadamard(values: torch.Tensor) -> torch.Tensor:
    width = int(values.shape[-1])
    block_size = rht_block_size(width)
    original_shape = values.shape
    work = values.reshape(-1, width // block_size, block_size).clone()
    butterfly = 1
    while butterfly < block_size:
        paired = work.reshape(-1, width // block_size, block_size // (2 * butterfly), 2, butterfly)
        left = paired[..., 0, :].clone()
        right = paired[..., 1, :].clone()
        paired[..., 0, :] = left + right
        paired[..., 1, :] = left - right
        butterfly *= 2
    return work.reshape(original_shape) * (1.0 / math.sqrt(block_size))


class _RandomizedBlockHadamard(torch.autograd.Function):
    """Memory-bounded orthogonal transform with a recomputed inverse."""

    @staticmethod
    def forward(ctx, values: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(signs)
        with torch.no_grad():
            signed = values * signs.to(device=values.device, dtype=values.dtype)
            return _normalized_block_hadamard(signed)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        (signs,) = ctx.saved_tensors
        with torch.no_grad():
            grad_input = _normalized_block_hadamard(grad_output)
            grad_input = grad_input * signs.to(device=grad_output.device, dtype=grad_output.dtype)
        return grad_input, None


def randomized_block_hadamard_transform(
    values: torch.Tensor,
    signs: torch.Tensor,
) -> torch.Tensor:
    """Apply ``D H / sqrt(block_size)`` to the final dimension."""

    if values.ndim == 0:
        raise ValueError("RHT input must have a final hidden dimension.")
    width = int(values.shape[-1])
    if tuple(signs.shape) != (width,):
        raise ValueError(f"RHT signs must have shape [{width}], got {tuple(signs.shape)}.")
    return _RandomizedBlockHadamard.apply(values, signs)
