"""Function-preserving local rotations for WAM collection/calibration.

Rotate both W and x, then calibrate D/gamma in that coordinate system.
Weights and rotated inputs are rounded to the model dtype before each Linear;
the native producer implements the same rounding boundary.
"""
from __future__ import annotations

import torch

from .rht import rht_signs, rht_transform
from .topology import enumerate_fastwamjoint_linears


def rotation_identity(rotation: str, seed: int) -> dict:
    if rotation not in ("none", "rht"):
        raise ValueError(f"Unsupported WAM rotation: {rotation}")
    return {} if rotation == "none" else {
        "rotation": rotation, "rotation_seed": int(seed),
        "rotation_contract": "rht_fp32_to_model_v1",
    }


@torch.no_grad()
def install_rht_(model, *, seed: int = 42) -> None:
    sites = enumerate_fastwamjoint_linears(model)
    for site in sites:
        module = site.module
        width = module.in_features
        if width not in (3072, 14336) and (width <= 0 or width & (width - 1)):
            raise ValueError(f"Unsupported official rotation width: {width}")
        if hasattr(module, "_wam_rotation_signs"):
            raise ValueError(f"Rotation already installed: {site.module_name}")
    for site in sites:
        module = site.module
        signs = rht_signs(module.in_features, seed=seed,
                             module_name=site.module_name, device=module.weight.device)
        # Chunk rows to avoid a second full FP32 copy of a wide FFN weight.
        for start in range(0, module.out_features, 128):
            rows = module.weight[start:start + 128]
            rows.copy_(rht_transform(rows.float(), signs, "rht").to(rows.dtype))
        module.register_buffer("_wam_rotation_signs", signs, persistent=False)
        module._wam_rotation = "rht"
        module._wam_rotation_config = rotation_identity("rht", seed)

        def rotate_input(linear, inputs):
            x, *rest = inputs
            rotated = rht_transform(x.float(), linear._wam_rotation_signs, "rht")
            return (rotated.to(x.dtype), *rest)

        # Must precede activation cache pre-hooks and sensitivity forward hooks.
        module.register_forward_pre_hook(rotate_input)
