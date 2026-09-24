"""Explicit, process-scoped robot runtime configuration; no model/data changes."""
import json
import os
from pathlib import Path

_configured_tile = None


def resolve_runtime_profile(*, profile=None, construct_device=None, cuda_graph=None, fuse_block=None):
    path = profile or os.environ.get("FASTWAM_RUNTIME_PROFILE")
    supplied = dict(construct_device=construct_device, cuda_graph=cuda_graph, fuse_block=fuse_block)
    if not path:
        defaults = dict(construct_device="meta", cuda_graph=False, fuse_block=False)
        return {k: defaults[k] if v is None else v for k, v in supplied.items()}, None
    path = Path(path).expanduser().resolve()
    spec = json.loads(path.read_text())
    required = {"schema_version", "name", "weight_bits", "activation_bits", "rotation",
                "construct_device", "cuda_graph", "fuse_block", "w4a8_tile"}
    if set(spec) != required:
        raise ValueError("Unexpected or missing runtime profile fields.")
    expected = dict(schema_version=1, weight_bits=4, activation_bits=8, rotation="none",
                    construct_device="cpu", cuda_graph=True, fuse_block=False, w4a8_tile=64)
    if any(type(spec[k]) is not type(v) or spec[k] != v for k, v in expected.items()):
        raise ValueError("Robot runtime profile requires W4A8, CPU construction, Graph, tile64 and no block fusion.")
    if not isinstance(spec['name'], str) or not spec['name'].strip():
        raise ValueError("Runtime profile needs a nonempty name.")
    for key, value in supplied.items():
        if value is not None and value != spec[key]:
            raise ValueError(f"Explicit {key}={value!r} conflicts with runtime profile {spec[key]!r}; update the server configuration.")
    configure_tile64()
    return {k: spec[k] for k in supplied}, {**spec, "path": str(path)}


def configure_tile64():
    global _configured_tile
    from .kernels.w4a8.loader import load_extension
    if os.environ.get("FASTWAM_W4A8_TILE") not in (None, "64"):
        raise ValueError("Runtime profile conflicts with FASTWAM_W4A8_TILE; restart with tile64.")
    if "FASTWAM_W4A8_EXPERIMENTAL_SMALL_TILE" in os.environ:
        raise ValueError("Unset FASTWAM_W4A8_EXPERIMENTAL_SMALL_TILE before using the robot profile.")
    if load_extension.cache_info().currsize and _configured_tile != "64":
        raise RuntimeError("W4A8 extension already loaded before runtime profile; restart the server process.")
    os.environ["FASTWAM_W4A8_TILE"] = "64"
    _configured_tile = "64"


def validate_profile_metadata(profile, metadata):
    if profile is None:
        return
    if (metadata.get("weight_bits") != 4 or metadata.get("activation_bits") != 8
            or not metadata.get("sites")
            or any(s["rotation"] != "none" for s in metadata["sites"])):
        raise ValueError("The selected runtime profile only supports non-rotated W4A8 deployments.")
