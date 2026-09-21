from __future__ import annotations

import json
import os
from pathlib import Path

import torch

from ..recovery import atomic_snapshot

FORMAT = "fastwam_steerquant_observations_v1"


def _portable(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().contiguous()
    if value is None or type(value) in (str, int, float, bool):
        return value
    if isinstance(value, dict) and all(isinstance(k, str) for k in value):
        return {k: _portable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_portable(v) for v in value)
    raise TypeError("Observation records must contain CPU tensors/primitive values; convert NumPy arrays to tensors.")


def validate_records(records):
    if not records:
        raise ValueError("Calibration observation set must be nonempty.")
    ids, indices = set(), set()
    for record in records:
        index, identity = record["sample_index"], record["observation_id"]
        if not isinstance(index, int) or index < 0 or not isinstance(identity, str) or not identity:
            raise ValueError("Each record needs a nonnegative sample_index and a nonempty observation_id.")
        if index in indices or identity in ids:
            raise ValueError("Duplicate observation ID or sample index.")
        indices.add(index); ids.add(identity)


def save_observation_records(records, path):
    records = _portable(list(records))
    validate_records(records)
    path = Path(path).expanduser().resolve()
    atomic_snapshot({"format": FORMAT, "records": records}, path)
    return path


def load_observation_records(path):
    payload = torch.load(Path(path).expanduser(), map_location="cpu", weights_only=True)
    if payload.get("format") != FORMAT:
        raise ValueError("Use the new observation format; explicitly adapt old preprocessing first.")
    records = list(payload["records"])
    validate_records(records)
    return records


def merge_observation_files(paths, output):
    records = [record for path in paths for record in load_observation_records(path)]
    return save_observation_records(sorted(records, key=lambda r: r["sample_index"]), output)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temporary, path)
