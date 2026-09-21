"""Bounded-memory, atomic progress snapshots for long calibration jobs."""
from __future__ import annotations

import ctypes
import gc
import os
from pathlib import Path

import torch


def file_identity(path) -> dict:
    path = Path(path).resolve()
    info = path.stat()
    return {"path": str(path), "size": info.st_size, "mtime_ns": info.st_mtime_ns}


def trim_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # Emptying CUDA's allocator does not release freed CPU offload buffers.
    try:
        ctypes.CDLL(None).malloc_trim(0)
    except (AttributeError, OSError):
        pass


def memory_status() -> str:
    values = []
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith(("VmRSS:", "VmHWM:")):
            key, amount, _unit = line.split()
            values.append(f"{key[:-1]}={int(amount) / 2**20:.2f}GiB")
    for name in ("memory.current", "memory.max"):
        path = Path("/sys/fs/cgroup") / name
        if path.exists():
            value = path.read_text().strip()
            values.append(f"cgroup_{name}={int(value) / 2**30:.2f}GiB" if value.isdigit() else f"{name}={value}")
    return " ".join(values)


def atomic_snapshot(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # A fixed pending filename bounds disk use even after SIGKILL. Callers
    # must hold an exclusive worker lock for the lifetime of this state.
    temporary = path.with_name(path.name + ".pending")
    with temporary.open("wb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def sensitivity_commit_due(completed: int, total: int, since_commit: int,
                           snapshot_every: int, recycle_due: bool) -> bool:
    """Never recycle or finish with uncommitted observations."""
    if snapshot_every <= 0:
        raise ValueError("snapshot_every must be positive.")
    return completed == total or since_commit >= snapshot_every or recycle_due


def sensitivity_memory_pressure(*, rss_limit_gib: float, cgroup_limit_gib: float,
                                rss_bytes: int | None = None,
                                cgroup_bytes: int | None = None) -> str | None:
    """Between-observation soft recycle trigger, not a hard OOM guarantee."""
    if rss_limit_gib < 0 or cgroup_limit_gib < 0:
        raise ValueError("Memory thresholds must be nonnegative (0 disables).")
    if not rss_limit_gib and not cgroup_limit_gib:
        return None
    if rss_bytes is None:
        for line in Path('/proc/self/status').read_text().splitlines():
            if line.startswith('VmRSS:'):
                rss_bytes = int(line.split()[1]) * 1024
                break
    if cgroup_bytes is None:
        path = Path('/sys/fs/cgroup/memory.current')
        if path.exists():
            cgroup_bytes = int(path.read_text())
    for label, value, limit in (("worker RSS", rss_bytes, rss_limit_gib),
                                ("cgroup memory", cgroup_bytes, cgroup_limit_gib)):
        if limit and value is not None and value >= limit * 2**30:
            return f"{label} {value / 2**30:.2f} GiB >= {limit:.2f} GiB"
    return None
