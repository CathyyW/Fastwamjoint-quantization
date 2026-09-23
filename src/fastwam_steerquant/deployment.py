"""Complete, CPU-serializable deployments. Native weights stay packed on load."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import torch
from torch import nn

from .checkpoint import QuantizationCheckpoint
from .kernels.w4a8.ops import pack_signed_int4
from .recovery import atomic_snapshot
from .runtime import WAMQuantLinear
from .state import DenoiseCallState
from .streams import FastWAMStreamConfig
from .topology import enumerate_fastwamjoint_linears, resolve_site_module, resolve_site_parent

FORMAT = "fastwam_steerquant_packed_deployment_v1"
BUFFERS = ("weight", "weight_scales", "input_scale", "gamma_gains", "clipping_ranges",
           "bias", "input_rotation_signs")


def _cpu(tensor):
    if tensor.is_meta:
        raise ValueError("Export requires materialized model weights.")
    return tensor.detach().cpu().contiguous()


def _target_aliases(model, names):
    """Find every registered path to each target, including MoT/dit aliases."""
    by_id = {}
    for name in names:
        key = id(resolve_site_module(model, name))
        if key in by_id:
            raise ValueError("Distinct calibration sites share one Linear; unsupported tying.")
        by_id[key] = name
    aliases = {name: [] for name in names}
    for path, module in model.named_modules(remove_duplicate=False):
        if id(module) in by_id:
            aliases[by_id[id(module)]].append(path)
    return {name: sorted(paths) for name, paths in aliases.items()}


def _spec_aliases(spec):
    # Old v1 files without aliases remain valid for non-aliased builders only.
    return spec.get("aliases", [spec["module_name"]])


def export_deployment(model: nn.Module, checkpoint: QuantizationCheckpoint, path: str | Path,
                      *, model_config: dict, provenance: dict | None = None) -> Path:
    """Export from the ORIGINAL model, not a rotated or already replaced model.

    Runs on CPU when the caller loads the original model on CPU. Original FP
    weights are not mutated; target weights are omitted from the output.
    """
    checkpoint.validate()
    if checkpoint.weight_bits != 4 or checkpoint.activation_bits not in (4, 8):
        raise ValueError("Deployment requires W4A8 or W4A4 calibration.")
    original_sites = enumerate_fastwamjoint_linears(model)
    if {s.module_name for s in original_sites} != {s.module_name for s in checkpoint.sites}:
        raise ValueError("Export requires calibration of every target Linear, not a partial shard.")
    if any(hasattr(s.module, "_wam_rotation") for s in original_sites):
        raise ValueError("Export requires the original, unrotated model.")
    # JSON metadata cannot serialize arbitrary Python objects or execute code on load.
    metadata = json.loads(json.dumps({"model_config": model_config, "provenance": provenance or {}}))
    aliases = _target_aliases(model, [s.module_name for s in checkpoint.sites])
    excluded = {alias + "." + name for paths in aliases.values()
                for alias in paths for name in ("weight", "bias")}
    state = {name: _cpu(value) for name, value in model.state_dict().items() if name not in excluded}
    specs = []
    for entry in checkpoint.sites:
        source = resolve_site_module(model, entry.module_name)
        if tuple(source.weight.shape) != tuple(entry.qweight.shape):
            raise ValueError(f"Shape mismatch: {entry.module_name}")
        n, k = entry.qweight.shape
        alignment = 256 if checkpoint.activation_bits == 4 else 32
        if k % alignment or n % 8:
            raise ValueError(f"Unsupported native shape {entry.module_name}: N={n}, K={k}")
        if entry.rotation != "none" and checkpoint.activation_bits != 4:
            raise ValueError("RHT requires W4A4.")
        spec = {"module_name": entry.module_name, "aliases": aliases[entry.module_name],
                "site_index": entry.site_index,
                "expert": entry.expert, "operation": entry.operation,
                "stream_names": list(entry.stream_names), "rotation": entry.rotation,
                "in_features": k, "out_features": n, "has_bias": source.bias is not None}
        values = {"weight": pack_signed_int4(entry.qweight.cpu()),
                  "weight_scales": entry.weight_scales.float().reshape(-1),
                  "input_scale": entry.input_scale.float().reciprocal(),
                  "gamma_gains": entry.gamma_gains.float(),
                  "clipping_ranges": entry.clipping_ranges.float() / (2 ** (checkpoint.activation_bits - 1) - 1),
                  "bias": source.bias, "input_rotation_signs": entry.input_rotation_signs}
        for name, value in values.items():
            if value is not None:
                # Reuse the exact tensor, so torch.save writes one storage even
                # though strict state_dict loading needs every registered path.
                packed_value = _cpu(value)
                for alias in aliases[entry.module_name]:
                    state[alias + "." + name] = packed_value
        specs.append(spec)
    # Include registered nonpersistent buffers (e.g. positional caches) so meta
    # construction does not leave an uninitialized buffer at model.to(device).
    nonpersistent = {}
    for prefix, module in model.named_modules():
        for name in module._non_persistent_buffers_set:
            value = module._buffers[name]
            if value is not None:
                nonpersistent[f"{prefix}.{name}" if prefix else name] = _cpu(value)
    payload = {"format": FORMAT, "weight_bits": 4, "activation_bits": checkpoint.activation_bits,
               "num_calls": checkpoint.num_calls, "stream_config": checkpoint.stream_config,
               "sites": specs, "state_dict": state, "nonpersistent_buffers": nonpersistent,
               "d_config": checkpoint.d_config, "gamma_config": checkpoint.gamma_config, **metadata}
    validate_payload(payload)
    output = Path(path).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Deployment already exists: {output}")
    atomic_snapshot(payload, output)
    return output


def validate_payload(payload: dict) -> None:
    if payload.get("format") != FORMAT or payload.get("weight_bits") != 4:
        raise ValueError("Unsupported deployment format.")
    bits, calls = payload["activation_bits"], payload["num_calls"]
    if bits not in (4, 8) or calls <= 0 or not payload["sites"]:
        raise ValueError("Invalid precision, schedule or empty deployment.")
    FastWAMStreamConfig(**payload["stream_config"]).validate()
    names = [s["module_name"] for s in payload["sites"]]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate deployment sites.")
    state = payload["state_dict"]
    seen_aliases = set()
    for spec in payload["sites"]:
        aliases = _spec_aliases(spec)
        if (not isinstance(aliases, list) or not aliases
                or any(not isinstance(a, str) or not a or any(not p for p in a.split(".")) for a in aliases)
                or len(set(aliases)) != len(aliases) or spec["module_name"] not in aliases
                or seen_aliases.intersection(aliases)):
            raise ValueError("Invalid or overlapping deployment aliases.")
        seen_aliases.update(aliases)
        prefix = spec["module_name"] + "."
        n, k, streams = spec["out_features"], spec["in_features"], len(spec["stream_names"])
        if k <= 0 or n <= 0 or k % (256 if bits == 4 else 32) or n % 8 or streams < 1:
            raise ValueError("Unsupported packed deployment shape.")
        weight = state[prefix + "weight"]
        if weight.dtype != torch.uint8 or tuple(weight.shape) != (n, k // 2):
            raise ValueError("Deployment weight must be packed uint8 [N,K/2].")
        for name, shape in (("weight_scales", (n,)), ("input_scale", (k,)),
                            ("gamma_gains", (calls, streams)), ("clipping_ranges", (calls,))):
            value = state[prefix + name]
            if value.dtype != torch.float32 or tuple(value.shape) != shape:
                raise ValueError(f"Invalid deployment buffer: {prefix}{name}")
            if not torch.isfinite(value).all() or (value <= 0).any():
                raise ValueError(f"Nonpositive/nonfinite deployment scales: {prefix}{name}")
        bias = state.get(prefix + "bias")
        if spec["has_bias"] != (bias is not None) or (bias is not None and (
                bias.shape != (n,) or bias.dtype not in (torch.float16, torch.bfloat16))):
            raise ValueError("Native deployment bias must match model FP16/BF16 precision.")
        rotation = spec["rotation"]
        signs = state.get(prefix + "input_rotation_signs")
        if rotation not in ("none", "rht") or (rotation == "rht" and bits != 4):
            raise ValueError("Invalid deployment RHT mode.")
        if (rotation == "none") != (signs is None):
            raise ValueError("RHT metadata/signs mismatch.")
        if signs is not None and (signs.shape != (k,) or signs.dtype != torch.float32
                                  or not torch.all(signs.abs() == 1)):
            raise ValueError("Invalid RHT signs.")
        for alias in aliases:
            for name in BUFFERS:
                reference = state.get(prefix + name)
                value = state.get(alias + "." + name)
                if value is reference:
                    continue
                if (not isinstance(value, torch.Tensor) or not isinstance(reference, torch.Tensor)
                        or value.dtype != reference.dtype or value.shape != reference.shape
                        or not torch.equal(value, reference)):
                    raise ValueError(f"Conflicting or missing deployment alias buffer: {alias}.{name}")


def load_deployment(path: str | Path, build_model: Callable, *, device="cuda",
                    construct_device="meta") -> tuple[nn.Module, DenoiseCallState, dict]:
    """build_model(model_config, device=...) must construct WITHOUT loading a ckpt.

    No original FP target weight is ever moved to GPU. CPU construction is an
    explicit alternative for models whose constructors do not support meta.
    Unregistered tensors/external assets are the adapter's responsibility.
    """
    if construct_device not in ("meta", "cpu"):
        raise ValueError("Construct on meta or CPU to avoid FP GPU allocation.")
    payload = torch.load(Path(path).expanduser(), map_location="cpu", weights_only=True, mmap=True)
    validate_payload(payload)
    model = build_model(payload["model_config"], device=construct_device)
    if any(t.device.type not in ("meta", "cpu") for t in (*model.parameters(), *model.buffers())):
        raise ValueError("Builder placed weights on GPU before packed replacement.")
    existing = enumerate_fastwamjoint_linears(model)
    if {s.module_name for s in existing} != {s["module_name"] for s in payload["sites"]}:
        raise ValueError("Deployment/model topology mismatch.")
    aliases = _target_aliases(model, [s.module_name for s in existing])
    for spec in payload["sites"]:
        if aliases[spec["module_name"]] != sorted(_spec_aliases(spec)):
            raise ValueError(f"Builder alias topology differs at {spec['module_name']}")
    state = DenoiseCallState(payload["num_calls"])
    streams = FastWAMStreamConfig(**payload["stream_config"])
    for spec in payload["sites"]:
        source = resolve_site_module(model, spec["module_name"])
        if (source.in_features, source.out_features, source.bias is not None) != (
                spec["in_features"], spec["out_features"], spec["has_bias"]):
            raise ValueError(f"Builder shape/bias differs at {spec['module_name']}")
        values = {name: payload["state_dict"].get(spec["module_name"] + "." + name) for name in BUFFERS}
        replacement = WAMQuantLinear.from_packed(spec, values, state=state,
                stream_config=streams, activation_bits=payload["activation_bits"])
        for alias in _spec_aliases(spec):
            parent, child = resolve_site_parent(model, alias)
            setattr(parent, child, replacement)
    model.load_state_dict(payload["state_dict"], strict=True, assign=True)
    for name, value in payload["nonpersistent_buffers"].items():
        prefix, _, child = name.rpartition(".")
        parent = resolve_site_module(model, prefix) if prefix else model
        if child not in parent._non_persistent_buffers_set:
            raise ValueError(f"Nonpersistent buffer contract changed: {name}")
        parent._buffers[child] = value
    if any(t.is_meta for t in (*model.parameters(), *model.buffers())):
        raise ValueError("Builder left unmaterialized buffers; export or rebuild them explicitly.")
    model.to(device=device).eval()
    metadata = {k: v for k, v in payload.items() if k not in ("state_dict", "nonpersistent_buffers")}
    return model, state, metadata
