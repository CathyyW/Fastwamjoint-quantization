from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn

from .state import DenoiseCallState
from .streams import FastWAMStreamConfig, resolve_stream_layout, rows_for_stream
from .topology import LinearSite


@dataclass(frozen=True, order=True)
class CacheKey:
    site_index: int
    call_index: int
    stream_index: int


@dataclass
class _Cell:
    values: torch.Tensor
    priorities: torch.Tensor
    channel_absmax: torch.Tensor
    seen_rows: int
    stream_name: str


class ActivationCache:
    """CPU activation reservoir keyed by concrete Linear, call, and stream."""

    def __init__(self, *, num_calls: int, max_rows_per_cell: int = 2048, seed: int = 0,
                 preserve_source_dtype: bool = False) -> None:
        if num_calls <= 0 or max_rows_per_cell <= 0:
            raise ValueError("num_calls and max_rows_per_cell must be positive.")
        self.num_calls = int(num_calls)
        self.max_rows_per_cell = int(max_rows_per_cell)
        self.preserve_source_dtype = bool(preserve_source_dtype)
        self.rotation_config: dict[str, Any] = {}
        self._generator = torch.Generator(device="cpu").manual_seed(int(seed))
        self._cells: dict[CacheKey, _Cell] = {}

    @property
    def keys(self) -> tuple[CacheKey, ...]:
        return tuple(sorted(self._cells))

    def add(self, key: CacheKey, values: torch.Tensor, *, stream_name: str) -> None:
        if not 0 <= key.call_index < self.num_calls:
            raise IndexError("Cache call index is outside the configured schedule.")
        rows = values.detach().reshape(-1, values.shape[-1]).cpu()
        if not self.preserve_source_dtype:
            rows = rows.float()
        if rows.numel() == 0 or not torch.isfinite(rows).all():
            raise ValueError("Cached activation rows must be non-empty and finite.")
        priorities = torch.rand(rows.shape[0], generator=self._generator)
        current = self._cells.get(key)
        if current is None:
            combined_values = rows
            combined_priorities = priorities
            channel_absmax = rows.float().abs().amax(dim=0)
            seen_rows = rows.shape[0]
        else:
            if current.values.shape[1] != rows.shape[1] or current.stream_name != stream_name:
                raise ValueError(f"Inconsistent cache cell metadata for {key}.")
            combined_values = torch.cat([current.values, rows], dim=0)
            combined_priorities = torch.cat([current.priorities, priorities], dim=0)
            channel_absmax = torch.maximum(current.channel_absmax, rows.float().abs().amax(dim=0))
            seen_rows = current.seen_rows + rows.shape[0]
        if combined_values.shape[0] > self.max_rows_per_cell:
            keep = combined_priorities.topk(self.max_rows_per_cell, sorted=False).indices
            combined_values = combined_values.index_select(0, keep)
            combined_priorities = combined_priorities.index_select(0, keep)
        self._cells[key] = _Cell(
            values=combined_values,
            priorities=combined_priorities,
            channel_absmax=channel_absmax,
            seen_rows=seen_rows,
            stream_name=stream_name,
        )

    def values(self, key: CacheKey) -> torch.Tensor:
        # Calibrate one cell in FP32, not the entire reservoir at once.
        return self._cells[key].values.float()

    def channel_absmax(self, key: CacheKey) -> torch.Tensor:
        return self._cells[key].channel_absmax

    def seen_rows(self, key: CacheKey) -> int:
        return self._cells[key].seen_rows

    def stream_name(self, key: CacheKey) -> str:
        return self._cells[key].stream_name

    def site_keys(self, site_index: int) -> tuple[CacheKey, ...]:
        return tuple(key for key in self.keys if key.site_index == int(site_index))

    def state_dict(self) -> dict[str, Any]:
        return {
            "format": ("fastwamjoint_activation_cache_rotated_v2" if self.rotation_config
                       else "fastwamjoint_activation_cache_v1"),
            "num_calls": self.num_calls,
            "max_rows_per_cell": self.max_rows_per_cell,
            "preserve_source_dtype": self.preserve_source_dtype,
            "rotation_config": self.rotation_config,
            "generator_state": self._generator.get_state(),
            "cells": [
                {
                    "key": (key.site_index, key.call_index, key.stream_index),
                    "values": cell.values,
                    "priorities": cell.priorities,
                    "channel_absmax": cell.channel_absmax,
                    "seen_rows": cell.seen_rows,
                    "stream_name": cell.stream_name,
                }
                for key, cell in sorted(self._cells.items())
            ],
        }

    @classmethod
    def merge(cls, caches: Iterable["ActivationCache"]) -> "ActivationCache":
        """Merge independently sampled reservoirs without rereading activations."""

        selected = tuple(caches)
        if not selected:
            raise ValueError("At least one activation cache is required.")
        first = selected[0]
        if any(
            cache.num_calls != first.num_calls
            or cache.max_rows_per_cell != first.max_rows_per_cell
            or cache.rotation_config != first.rotation_config
            for cache in selected[1:]
        ):
            raise ValueError("Activation cache configurations do not match.")
        merged = cls(
            num_calls=first.num_calls,
            max_rows_per_cell=first.max_rows_per_cell,
            preserve_source_dtype=all(cache.preserve_source_dtype for cache in selected),
        )
        all_keys = sorted({key for cache in selected for key in cache.keys})
        merged.rotation_config = dict(first.rotation_config)
        for key in all_keys:
            cells = [cache._cells[key] for cache in selected if key in cache._cells]
            names = {cell.stream_name for cell in cells}
            widths = {cell.values.shape[1] for cell in cells}
            if len(names) != 1 or len(widths) != 1:
                raise ValueError(f"Inconsistent cache cell metadata for {key}.")
            values = torch.cat([cell.values for cell in cells], dim=0)
            priorities = torch.cat([cell.priorities for cell in cells], dim=0)
            if values.shape[0] > first.max_rows_per_cell:
                keep = priorities.topk(first.max_rows_per_cell, sorted=False).indices
                values = values.index_select(0, keep)
                priorities = priorities.index_select(0, keep)
            merged._cells[key] = _Cell(
                values=values,
                priorities=priorities,
                channel_absmax=torch.stack([cell.channel_absmax for cell in cells]).amax(dim=0),
                seen_rows=sum(cell.seen_rows for cell in cells),
                stream_name=cells[0].stream_name,
            )
        return merged

    def save(self, path: str | Path) -> Path:
        resolved = Path(path).expanduser().resolve()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        temporary = resolved.with_name(f".{resolved.name}.{os.getpid()}.tmp")
        torch.save(self.state_dict(), temporary)
        os.replace(temporary, resolved)
        return resolved

    @classmethod
    def load(cls, path: str | Path) -> "ActivationCache":
        try:
            payload = torch.load(Path(path).expanduser(), map_location="cpu", weights_only=True)
        except TypeError:
            payload = torch.load(Path(path).expanduser(), map_location="cpu")
        return cls.from_state_dict(payload)

    @classmethod
    def from_state_dict(cls, payload: dict[str, Any]) -> "ActivationCache":
        if payload.get("format") not in ("fastwamjoint_activation_cache_v1", "fastwamjoint_activation_cache_rotated_v2"):
            raise ValueError("Unsupported activation cache format.")
        if bool(payload.get("rotation_config")) != (payload["format"] == "fastwamjoint_activation_cache_rotated_v2"):
            raise ValueError("Activation cache format and rotation metadata disagree.")
        cache = cls(
            num_calls=int(payload["num_calls"]),
            max_rows_per_cell=int(payload["max_rows_per_cell"]),
            preserve_source_dtype=bool(payload.get("preserve_source_dtype", False)),
        )
        if "generator_state" in payload:
            cache._generator.set_state(payload["generator_state"])
        cache.rotation_config = dict(payload.get("rotation_config", {}))
        for stored in payload["cells"]:
            key = CacheKey(*(int(value) for value in stored["key"]))
            cache._cells[key] = _Cell(
                values=(stored["values"] if cache.preserve_source_dtype
                        else torch.as_tensor(stored["values"], dtype=torch.float32)),
                priorities=torch.as_tensor(stored["priorities"], dtype=torch.float32),
                channel_absmax=torch.as_tensor(stored["channel_absmax"], dtype=torch.float32),
                seen_rows=int(stored["seen_rows"]),
                stream_name=str(stored["stream_name"]),
            )
        return cache


class ActivationCollector:
    def __init__(
        self,
        sites: Iterable[LinearSite],
        *,
        state: DenoiseCallState,
        stream_config: FastWAMStreamConfig,
        cache: ActivationCache,
    ) -> None:
        self.sites = tuple(sites)
        self.state = state
        self.stream_config = stream_config
        self.cache = cache
        self._handles: list[Any] = []

    def _hook(self, site: LinearSite):
        def collect(_module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                raise TypeError(f"{site.module_name} did not receive a Tensor input.")
            x = inputs[0]
            layout = resolve_stream_layout(site, x.shape[-2], self.stream_config)
            call = self.state.require()
            for stream, (name, stream_slice) in enumerate(zip(layout.names, layout.slices())):
                self.cache.add(
                    CacheKey(site.index, call, stream),
                    rows_for_stream(x, stream_slice),
                    stream_name=name,
                )

        return collect

    def install(self) -> None:
        if self._handles:
            raise RuntimeError("Activation collector is already installed.")
        self._handles = [site.module.register_forward_pre_hook(self._hook(site)) for site in self.sites]

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def __enter__(self) -> "ActivationCollector":
        self.install()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.remove()
