from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch.nn as nn


ExpertName = Literal["video", "action"]
StreamKind = Literal["expert_tokens", "context"]

TARGET_LINEAR_PATHS: tuple[str, ...] = (
    "self_attn.q",
    "self_attn.k",
    "self_attn.v",
    "self_attn.o",
    "cross_attn.q",
    "cross_attn.k",
    "cross_attn.v",
    "cross_attn.o",
    "ffn.0",
    "ffn.2",
)


@dataclass(frozen=True)
class LinearSite:
    index: int
    expert: ExpertName
    block_index: int
    operation: str
    module_name: str
    # Calibration/enumeration sites carry the concrete Linear.  Runtime-only
    # metadata may omit it so replacing a Linear does not keep its BF16 weight
    # alive solely for stream-layout dispatch.
    module: nn.Linear | None
    stream_kind: StreamKind


def _resolve(root: nn.Module, path: str) -> nn.Module:
    module = root
    for part in path.split("."):
        module = module[int(part)] if part.isdigit() else getattr(module, part)
    return module


def enumerate_fastwamjoint_linears(model: nn.Module) -> tuple[LinearSite, ...]:
    """Enumerate the 10 W+A Linear sites in every video/action MoT block."""

    sites: list[LinearSite] = []
    for expert_name in ("video", "action"):
        expert = getattr(model, f"{expert_name}_expert", None)
        if expert is None or not hasattr(expert, "blocks"):
            raise ValueError(f"Model does not expose `{expert_name}_expert.blocks`.")
        for block_index, block in enumerate(expert.blocks):
            for operation in TARGET_LINEAR_PATHS:
                module = _resolve(block, operation)
                if not isinstance(module, nn.Linear):
                    raise TypeError(
                        f"{expert_name} block {block_index} `{operation}` is "
                        f"{type(module).__name__}, expected nn.Linear."
                    )
                sites.append(
                    LinearSite(
                        index=len(sites),
                        expert=expert_name,
                        block_index=block_index,
                        operation=operation,
                        module_name=f"{expert_name}_expert.blocks.{block_index}.{operation}",
                        module=module,
                        stream_kind="context" if operation in {"cross_attn.k", "cross_attn.v"} else "expert_tokens",
                    )
                )
    return tuple(sites)


def resolve_site_module(model: nn.Module, module_name: str) -> nn.Module:
    return _resolve(model, module_name)


def resolve_site_parent(model: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    parent_name, _, child_name = module_name.rpartition(".")
    return _resolve(model, parent_name), child_name
