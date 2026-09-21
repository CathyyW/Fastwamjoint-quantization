from __future__ import annotations

import hashlib
import importlib
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

from ..streams import FastWAMStreamConfig
from ..recovery import file_identity


def read_config(path) -> dict:
    """JSON paths in assets are relative to the config file, not the cwd."""
    path = Path(path).expanduser().resolve()
    config = json.loads(path.read_text())
    for key in ("adapter", "num_calls", "stream_config", "expected_sites"):
        if key not in config:
            raise ValueError(f"Missing configuration field: {key}")
    if config["num_calls"] <= 0 or config["expected_sites"] <= 0:
        raise ValueError("num_calls and expected_sites must be positive.")
    FastWAMStreamConfig(**config["stream_config"]).validate()
    for key, value in config.get("assets", {}).items():
        p = Path(value).expanduser()
        config["assets"][key] = str((path.parent / p).resolve() if not p.is_absolute() else p.resolve())
    return config


def _factory(config):
    module_name, separator, symbol = config["adapter"].partition(":")
    if not separator:
        raise ValueError("adapter must be a module:factory import path.")
    module = importlib.import_module(module_name)
    return getattr(module, symbol)


def config_identity(config) -> dict:
    """Guard resume against changed settings/assets/adapter implementation.

    Large asset files use path/size/mtime identity; archive content hashes in
    config['provenance'] for stronger cross-server provenance when available.
    List all external config/statistics/weights files under assets.
    """
    identity = {"config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
                "assets": {k: file_identity(v) for k, v in config.get("assets", {}).items()}}
    source = inspect.getsourcefile(_factory(config))
    if source:
        identity["adapter_sha256"] = hashlib.sha256(Path(source).read_bytes()).hexdigest()
    return identity


def load_adapter(config):
    config = read_config(config) if not isinstance(config, dict) else config
    adapter = _factory(config)(config)
    for name in ("load_model", "build_model", "prepare_inputs", "infer_kwargs", "action_scale"):
        if not callable(getattr(adapter, name, None)):
            raise TypeError(f"Adapter must implement {name}.")
    return adapter


@dataclass
class CallbackAdapter:
    """Connect existing robot code without changing the quantization algorithm.

    load_model(device=...) -> ORIGINAL trained model, eval mode
    build_model(model_config, device=...) -> structure only, no checkpoint load
    prepare_inputs(model, record, seed=...) -> five differentiable-denoise tensors
    infer_kwargs(model, record, seed=...) -> kwargs for production infer_action
    action_scale() -> positive finite std in MODEL-NORMALIZED action coordinates
    finalize_model(model, device=...) -> rebuild device-dependent non-state assets
    dataset_records() -> iterable of standard records (optional raw-data export)
    """
    load_model: Callable
    build_model: Callable
    prepare_inputs: Callable
    infer_kwargs: Callable
    action_scale: Callable
    finalize_model: Callable | None = None
    dataset_records: Callable | None = None


def validate_action_scale(value, *, device):
    value = torch.as_tensor(value, dtype=torch.float32, device=device)
    if value.ndim != 1 or value.numel() == 0 or not torch.isfinite(value).all() or (value <= 0).any():
        raise ValueError("action_scale must be a positive finite vector in normalized action coordinates.")
    return value
