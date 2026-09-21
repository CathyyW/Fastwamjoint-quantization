"""Matched latency measurements and explicit real-robot trial accounting."""
from __future__ import annotations
import json
import math
import statistics
import time
from pathlib import Path
import torch


def assert_finite(value):
    if isinstance(value, torch.Tensor):
        if not torch.isfinite(value).all():
            raise ValueError("Nonfinite inference output.")
    elif isinstance(value, dict):
        for item in value.values():
            assert_finite(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            assert_finite(item)


def latency_stats(samples):
    values = sorted(float(x) for x in samples)
    if not values or any(not math.isfinite(x) or x <= 0 for x in values):
        raise ValueError("Latency samples must be positive and finite.")
    return {"count": len(values), "mean_ms": statistics.mean(values),
            "p50_ms": statistics.median(values), "p95_ms": values[math.ceil(.95*len(values))-1]}


def measure_latency(call, *, warmup=3, repeats=20, device="cuda"):
    if warmup < 1 or repeats < 1:
        raise ValueError("Warmup and repeats must be positive.")
    device = torch.device(device)
    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    with torch.inference_mode():
        sync(); start = time.perf_counter()
        for _ in range(warmup):
            output = call()
            assert_finite(output)
            del output
        sync(); warmup_ms = (time.perf_counter()-start)*1000
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        samples = []
        for _ in range(repeats):
            sync(); start = time.perf_counter()
            output = call()
            sync(); samples.append((time.perf_counter()-start)*1000)
            assert_finite(output)
            del output
    return {**latency_stats(samples), "samples_ms": samples, "warmup_total_ms": warmup_ms,
            "steady_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
            "steady_peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None}


class TrialRecorder:
    """Append only finalized outcomes; never infers success from model predictions.

    Call once for EVERY attempted trial, including timeouts/aborts. This is a
    single-writer log intended for an existing robot experiment loop.
    """
    def __init__(self, path, *, mode, protocol_id, checkpoint_id):
        self.path = Path(path)
        self.metadata = dict(mode=mode, protocol_id=protocol_id, checkpoint_id=checkpoint_id)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, *, task, trial_id, success, reason, latencies_ms, seed=None):
        if type(success) is not bool or not task or not str(trial_id) or not reason:
            raise ValueError("Explicit task/trial ID, boolean success and terminal reason required.")
        values = [float(x) for x in latencies_ms]
        if values:
            latency_stats(values)
        record = {**self.metadata, "task": task, "trial_id": str(trial_id), "success": success,
                  "reason": reason, "latencies_ms": values, "seed": seed, "time_unix": time.time()}
        with self.path.open("a") as handle:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
            handle.flush()


def summarize_trials(paths):
    groups, seen = {}, set()
    for path in paths:
        for line in Path(path).read_text().splitlines():
            row = json.loads(line)
            if type(row["success"]) is not bool:
                raise ValueError("Each trial must have an explicit boolean outcome.")
            key = (row["protocol_id"], row["task"], row["mode"], row["checkpoint_id"])
            trial = (*key, row["trial_id"])
            if trial in seen:
                raise ValueError(f"Duplicate trial: {trial}")
            seen.add(trial)
            groups.setdefault(key, []).append(row)
    results = []
    for key, rows in sorted(groups.items()):
        latencies = [x for row in rows for x in row["latencies_ms"]]
        successes = sum(r["success"] for r in rows)
        results.append(dict(zip(("protocol_id", "task", "mode", "checkpoint_id"), key),
                            trials=len(rows), successes=successes, sr=successes/len(rows),
                            latency=latency_stats(latencies) if latencies else None))
    return results
