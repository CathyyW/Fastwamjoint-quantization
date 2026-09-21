from __future__ import annotations

import math
import os
import weakref
from dataclasses import dataclass, field as dataclass_field
from pathlib import Path
from typing import Callable, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .quant import fake_quant_activation_per_tensor, fake_quant_weight_per_channel
from .cache import ActivationCache, CacheKey
from .state import DenoiseCallState
from .streams import FastWAMStreamConfig, resolve_stream_layout, rows_for_stream
from .topology import LinearSite


@dataclass(frozen=True)
class SiteSensitivity:
    site_index: int
    module_name: str
    stream_names: tuple[str, ...]
    values: torch.Tensor
    observation_counts: torch.Tensor

    def validate(self, num_calls: int) -> None:
        expected = (num_calls, len(self.stream_names))
        if self.values.shape != expected or self.observation_counts.shape != expected:
            raise ValueError(f"Sensitivity tensors for {self.module_name} must have shape {expected}.")
        if not torch.isfinite(self.values).all() or (self.values < 0).any():
            raise ValueError("Sensitivity values must be finite and non-negative.")
        if (self.observation_counts <= 0).any():
            raise ValueError("Every sensitivity cell must contain an observation.")


@dataclass(frozen=True)
class SensitivityField:
    num_calls: int
    sites: tuple[SiteSensitivity, ...]
    rotation_config: dict = dataclass_field(default_factory=dict)

    def validate(self) -> None:
        if self.num_calls <= 0 or not self.sites:
            raise ValueError("Sensitivity field must contain calls and sites.")
        indices = [site.site_index for site in self.sites]
        if len(indices) != len(set(indices)):
            raise ValueError("Sensitivity site indices must be unique.")
        for site in self.sites:
            site.validate(self.num_calls)

    def by_site(self, site_index: int) -> SiteSensitivity:
        return next(site for site in self.sites if site.site_index == int(site_index))

    def state_dict(self) -> dict:
        self.validate()
        return {
            "format": ("fastwamjoint_sensitivity_rotated_v2" if self.rotation_config
                       else "fastwamjoint_sensitivity_v1"),
            "rotation_config": self.rotation_config,
            "num_calls": self.num_calls,
            "sites": [
                {
                    "site_index": site.site_index,
                    "module_name": site.module_name,
                    "stream_names": site.stream_names,
                    "values": site.values.cpu(),
                    "observation_counts": site.observation_counts.cpu(),
                }
                for site in self.sites
            ],
        }

    @classmethod
    def merge(cls, fields: Iterable["SensitivityField"]) -> "SensitivityField":
        """Merge RMS fields using their per-cell observation counts."""

        selected = tuple(fields)
        if not selected:
            raise ValueError("At least one sensitivity field is required.")
        num_calls = selected[0].num_calls
        if any(field.num_calls != num_calls for field in selected[1:]):
            raise ValueError("Sensitivity fields use different call counts.")
        if any(field.rotation_config != selected[0].rotation_config for field in selected[1:]):
            raise ValueError("Sensitivity fields use different rotation domains/seeds.")
        by_field = [{site.site_index: site for site in field.sites} for field in selected]
        indices = set(by_field[0])
        if any(set(items) != indices for items in by_field[1:]):
            raise ValueError("Sensitivity fields contain different site sets.")
        sites = []
        for site_index in sorted(indices):
            entries = [items[site_index] for items in by_field]
            metadata = {(item.module_name, item.stream_names) for item in entries}
            if len(metadata) != 1:
                raise ValueError(f"Sensitivity metadata differs for site {site_index}.")
            counts = torch.stack([item.observation_counts for item in entries]).sum(dim=0)
            square_sum = torch.stack(
                [item.values.square() * item.observation_counts for item in entries]
            ).sum(dim=0)
            sites.append(
                SiteSensitivity(
                    site_index=site_index,
                    module_name=entries[0].module_name,
                    stream_names=entries[0].stream_names,
                    values=(square_sum / counts.clamp_min(1)).sqrt(),
                    observation_counts=counts,
                )
            )
        result = cls(num_calls=num_calls, sites=tuple(sites), rotation_config=dict(selected[0].rotation_config))
        result.validate()
        return result

    def save(self, path: str | Path) -> Path:
        resolved = Path(path).expanduser().resolve()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        temporary = resolved.with_name(f".{resolved.name}.{os.getpid()}.tmp")
        torch.save(self.state_dict(), temporary)
        os.replace(temporary, resolved)
        return resolved

    @classmethod
    def load(cls, path: str | Path) -> "SensitivityField":
        try:
            payload = torch.load(Path(path).expanduser(), map_location="cpu", weights_only=True)
        except TypeError:
            payload = torch.load(Path(path).expanduser(), map_location="cpu")
        if payload.get("format") not in ("fastwamjoint_sensitivity_v1", "fastwamjoint_sensitivity_rotated_v2"):
            raise ValueError("Unsupported sensitivity field format.")
        if bool(payload.get("rotation_config")) != (payload["format"] == "fastwamjoint_sensitivity_rotated_v2"):
            raise ValueError("Sensitivity format and rotation metadata disagree.")
        field = cls(
            num_calls=int(payload["num_calls"]),
            rotation_config=dict(payload.get("rotation_config", {})),
            sites=tuple(
                SiteSensitivity(
                    site_index=int(item["site_index"]),
                    module_name=str(item["module_name"]),
                    stream_names=tuple(item["stream_names"]),
                    values=torch.as_tensor(item["values"], dtype=torch.float32),
                    observation_counts=torch.as_tensor(item["observation_counts"], dtype=torch.long),
                )
                for item in payload["sites"]
            ),
        )
        field.validate()
        return field


class SensitivityAccumulator:
    """RMS-aggregate observation-level action sensitivities."""

    def __init__(self, *, num_calls: int) -> None:
        self.num_calls = int(num_calls)
        self._sum_squares: dict[int, torch.Tensor] = {}
        self._counts: dict[int, torch.Tensor] = {}
        self._metadata: dict[int, tuple[str, tuple[str, ...]]] = {}

    def add(
        self,
        site: LinearSite,
        stream_names: tuple[str, ...],
        values: torch.Tensor,
    ) -> None:
        values = torch.as_tensor(values, dtype=torch.float32).cpu()
        expected = (self.num_calls, len(stream_names))
        if values.shape != expected or not torch.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"Observation sensitivity must be finite, non-negative, and shaped {expected}.")
        metadata = (site.module_name, stream_names)
        if site.index in self._metadata and self._metadata[site.index] != metadata:
            raise ValueError("Sensitivity stream layout changed across observations.")
        self._metadata[site.index] = metadata
        self._sum_squares[site.index] = self._sum_squares.get(site.index, torch.zeros_like(values)) + values.square()
        self._counts[site.index] = self._counts.get(site.index, torch.zeros_like(values, dtype=torch.long)) + 1

    def state_dict(self) -> dict:
        return {"num_calls": self.num_calls, "sum_squares": self._sum_squares,
                "counts": self._counts, "metadata": self._metadata}

    @classmethod
    def from_state_dict(cls, payload: dict) -> "SensitivityAccumulator":
        result = cls(num_calls=int(payload["num_calls"]))
        result._sum_squares = payload["sum_squares"]
        result._counts = payload["counts"]
        result._metadata = payload["metadata"]
        return result

    def finalize(self, *, rotation_config: dict | None = None) -> SensitivityField:
        entries = []
        for site_index in sorted(self._metadata):
            module_name, stream_names = self._metadata[site_index]
            counts = self._counts[site_index]
            entries.append(
                SiteSensitivity(
                    site_index=site_index,
                    module_name=module_name,
                    stream_names=stream_names,
                    values=(self._sum_squares[site_index] / counts).sqrt(),
                    observation_counts=counts,
                )
            )
        field = SensitivityField(self.num_calls, tuple(entries), dict(rotation_config or {}))
        field.validate()
        return field


@dataclass
class _Invocation:
    site: LinearSite
    call: int
    x: torch.Tensor
    y: torch.Tensor


def estimate_action_sensitivity(
    run: Callable[[], torch.Tensor],
    sites: Iterable[LinearSite],
    *,
    state: DenoiseCallState,
    stream_config: FastWAMStreamConfig,
    action_scale: torch.Tensor | float,
    weight_bits: int,
    activation_bits: int,
    num_probes: int = 4,
    seed: int = 0,
) -> dict[int, tuple[tuple[str, ...], torch.Tensor]]:
    """Estimate ||D_a^-1 J P_s delta_y||_RMS using Rademacher VJPs.

    `run` must execute a differentiable full denoising trajectory and return the
    final action tensor without detaching it. Use only a small site group per
    call to keep the retained graph bounded.
    """

    selected = tuple(sites)
    if not selected or num_probes <= 0:
        raise ValueError("At least one site and one probe are required.")
    invocations: list[_Invocation] = []
    handles = []

    def make_hook(site: LinearSite):
        def capture(_module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            if not inputs or not isinstance(inputs[0], torch.Tensor) or not isinstance(output, torch.Tensor):
                raise TypeError("Sensitivity hooks require Tensor Linear inputs and outputs.")
            invocations.append(_Invocation(site, state.require(), inputs[0], output))

        return capture

    for site in selected:
        handles.append(site.module.register_forward_hook(make_hook(site)))
    try:
        final_action = run()
    finally:
        for handle in handles:
            handle.remove()
    if not final_action.requires_grad:
        raise RuntimeError("Final action is detached; run a differentiable denoising trajectory.")

    keyed: dict[tuple[int, int], _Invocation] = {}
    for invocation in invocations:
        key = (invocation.site.index, invocation.call)
        if key in keyed:
            raise RuntimeError(f"{invocation.site.module_name} ran more than once in denoise call {invocation.call}.")
        keyed[key] = invocation
    expected = {(site.index, call) for site in selected for call in range(state.num_calls)}
    missing = expected.difference(keyed)
    if missing:
        raise RuntimeError(f"Missing {len(missing)} selected Linear/call invocations.")

    outputs = [keyed[key].y for key in sorted(keyed)]
    residuals: dict[tuple[int, int], torch.Tensor] = {}
    layouts = {}
    for key, invocation in keyed.items():
        x = invocation.x.detach()
        y = invocation.y.detach()
        qx = fake_quant_activation_per_tensor(x, bits=activation_bits)
        qw = fake_quant_weight_per_channel(invocation.site.module.weight.detach(), bits=weight_bits)
        bias = invocation.site.module.bias
        residuals[key] = F.linear(qx, qw, None if bias is None else bias.detach()) - y
        layouts[key] = resolve_stream_layout(invocation.site, x.shape[-2], stream_config)

    generator = torch.Generator(device=final_action.device).manual_seed(int(seed))
    normalized = final_action.float() / torch.as_tensor(
        action_scale, device=final_action.device, dtype=torch.float32
    ).clamp_min(1e-8)
    accum: dict[tuple[int, int, int], float] = {}
    for probe_index in range(num_probes):
        probe = torch.empty_like(normalized).bernoulli_(0.5, generator=generator).mul_(2).sub_(1)
        scalar = (normalized * probe).sum() / math.sqrt(normalized.numel())
        gradients = torch.autograd.grad(
            scalar,
            outputs,
            retain_graph=probe_index + 1 < num_probes,
            allow_unused=False,
        )
        for key, gradient in zip(sorted(keyed), gradients):
            layout = layouts[key]
            residual = residuals[key]
            for stream, stream_slice in enumerate(layout.slices()):
                impact = (gradient[..., stream_slice, :].float() * residual[..., stream_slice, :].float()).sum()
                cell = (key[0], key[1], stream)
                accum[cell] = accum.get(cell, 0.0) + float(impact.detach().square().cpu())

    result: dict[int, tuple[tuple[str, ...], torch.Tensor]] = {}
    for site in selected:
        first_layout = layouts[(site.index, 0)]
        values = torch.zeros(state.num_calls, first_layout.num_streams)
        for call in range(state.num_calls):
            layout = layouts[(site.index, call)]
            if layout.names != first_layout.names:
                raise ValueError("Stream names changed across denoising calls.")
            for stream in range(layout.num_streams):
                values[call, stream] = math.sqrt(accum[(site.index, call, stream)] / num_probes)
        result[site.index] = (first_layout.names, values)
    return result


class LazyActionSensitivityCollector:
    """Score all Linears in one reverse pass by constructing residuals lazily."""

    def __init__(
        self,
        sites: Iterable[LinearSite],
        *,
        state: DenoiseCallState,
        stream_config: FastWAMStreamConfig,
        weight_bits: int,
        activation_bits: int,
        offload_captures: bool = True,
        accumulate_on_device: bool = False,
        deduplicate_captures: bool = False,
    ) -> None:
        self.sites = tuple(sites)
        self.state = state
        self.stream_config = stream_config
        self.weight_bits = int(weight_bits)
        self.activation_bits = int(activation_bits)
        self.offload_captures = bool(offload_captures)
        self.accumulate_on_device = bool(accumulate_on_device)
        self.deduplicate_captures = bool(deduplicate_captures)
        self._input_offloads = {}
        self.activation_cache: ActivationCache | None = None
        self._device_sum_squares: dict[tuple[int, int], torch.Tensor] = {}
        self._handles: list[object] = []
        self._captures: dict[tuple[int, int], tuple[LinearSite, torch.Tensor, torch.Tensor]] = {}
        self._stream_names: dict[int, tuple[str, ...]] = {}
        self._sum_squares: dict[tuple[int, int, int], float] = {}
        self._quantized_weights: dict[int, torch.Tensor] = {}

    def _offload_input(self, tensor: torch.Tensor) -> torch.Tensor:
        if not self.deduplicate_captures:
            return tensor.detach().to(device="cpu", non_blocking=False)
        # Reuse only the SAME live Tensor object with the SAME mutation version.
        # No data_ptr-only key (allocator reuse) or value-based approximation.
        key = id(tensor)
        version = tensor._version
        previous = self._input_offloads.get(key)
        if previous is not None and previous[0]() is tensor and previous[1] == version:
            return previous[2]
        host = tensor.detach().to(device="cpu", non_blocking=False)
        self._input_offloads[key] = (weakref.ref(tensor), version, host)
        return host

    def _forward_hook(
        self,
        site: LinearSite,
        _module: nn.Module,
        inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        if not inputs or not isinstance(inputs[0], torch.Tensor) or not isinstance(output, torch.Tensor):
            raise TypeError("Sensitivity hooks require Tensor Linear inputs and outputs.")
        key = (site.index, self.state.require())
        if key in self._captures:
            raise RuntimeError(f"{site.module_name} ran more than once in denoise call {key[1]}.")
        if not output.requires_grad:
            output.requires_grad_(True)
        x = inputs[0].detach()
        y = output.detach()
        layout = resolve_stream_layout(site, x.shape[-2], self.stream_config)
        previous = self._stream_names.setdefault(site.index, layout.names)
        if previous != layout.names:
            raise ValueError(f"Stream layout changed for {site.module_name}.")
        if self.offload_captures:
            x = self._offload_input(inputs[0])
            y = y.to(device="cpu", non_blocking=False)
        self._captures[key] = (site, x, y)
        # Reuse the exact detached input already copied for sensitivity. This
        # avoids a second D2H copy by a separate activation pre-hook.
        if self.activation_cache is not None:
            for stream, (name, stream_slice) in enumerate(zip(layout.names, layout.slices())):
                self.activation_cache.add(CacheKey(site.index, key[1], stream),
                                          rows_for_stream(x, stream_slice), stream_name=name)
        output.register_hook(
            lambda gradient, selected=site, selected_key=key, selected_layout=layout: self._backward_hook(
                selected, selected_key, selected_layout, gradient
            )
        )

    def _backward_hook(self, site, key, layout, gradient: torch.Tensor) -> torch.Tensor:
        _site, x, y = self._captures.pop(key)
        if self.offload_captures:
            x = x.to(device=gradient.device, non_blocking=False)
            y = y.to(device=gradient.device, non_blocking=False)
        quantized_weight = self._quantized_weights.get(site.index)
        if quantized_weight is None:
            quantized_weight = fake_quant_weight_per_channel(
                site.module.weight.detach(), bits=self.weight_bits
            )
            self._quantized_weights[site.index] = quantized_weight
        quantized_x = fake_quant_activation_per_tensor(x, bits=self.activation_bits)
        bias = site.module.bias
        residual = F.linear(
            quantized_x,
            quantized_weight,
            None if bias is None else bias.detach(),
        ) - y
        squares = []
        for stream_index, stream_slice in enumerate(layout.slices()):
            directional = (
                gradient[..., stream_slice, :].float()
                * residual[..., stream_slice, :].float()
            ).sum()
            squared = directional.detach().square()
            if self.accumulate_on_device:
                squares.append(squared)
            else:
                cell = (site.index, key[1], stream_index)
                self._sum_squares[cell] = self._sum_squares.get(cell, 0.0) + float(squared.cpu())
        if self.accumulate_on_device:
            # Original: FP32 square -> Python double sum in probe order.
            # Keep that rounding/order, with just one host transfer at finish.
            values = torch.stack(squares).double()
            if key in self._device_sum_squares:
                self._device_sum_squares[key].add_(values)
            else:
                self._device_sum_squares[key] = values
        return gradient

    def install(self) -> None:
        if self._handles:
            raise RuntimeError("Sensitivity collector is already installed.")
        self._handles = [
            site.module.register_forward_hook(
                lambda module, inputs, output, selected=site: self._forward_hook(
                    selected, module, inputs, output
                )
            )
            for site in self.sites
        ]

    def begin_probe(self) -> None:
        self._captures.clear()
        self._input_offloads.clear()

    def validate_forward(self) -> None:
        # Captures now own their host references. Do not retain all inputs
        # throughout backward after individual captures have been consumed.
        self._input_offloads.clear()
        expected = {
            (site.index, call)
            for site in self.sites
            for call in range(self.state.num_calls)
        }
        missing = expected.difference(self._captures)
        if missing:
            first = min(missing)
            raise RuntimeError(
                f"Missing {len(missing)} Linear/call captures; first site/call={first}."
            )

    def finish_observation(self, num_probes: int) -> dict[int, tuple[tuple[str, ...], torch.Tensor]]:
        if num_probes <= 0:
            raise ValueError("num_probes must be positive.")
        if self.accumulate_on_device and self._device_sum_squares:
            keys = sorted(self._device_sum_squares)
            flat = torch.cat([self._device_sum_squares[key] for key in keys]).cpu().tolist()
            offset = 0
            for site_index, call in keys:
                for stream in range(len(self._stream_names[site_index])):
                    self._sum_squares[(site_index, call, stream)] = flat[offset]
                    offset += 1
        result = {}
        for site in self.sites:
            names = self._stream_names[site.index]
            values = torch.zeros(self.state.num_calls, len(names))
            for call in range(self.state.num_calls):
                for stream in range(len(names)):
                    values[call, stream] = math.sqrt(
                        self._sum_squares.get((site.index, call, stream), 0.0) / num_probes
                    )
            result[site.index] = (names, values)
        return result

    def reset_observation(self) -> None:
        self._sum_squares.clear()
        self._device_sum_squares.clear()
        self.activation_cache = None
        self._stream_names.clear()
        self.begin_probe()

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self.reset_observation()
        self._quantized_weights.clear()

    def __enter__(self) -> "LazyActionSensitivityCollector":
        self.install()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.remove()
