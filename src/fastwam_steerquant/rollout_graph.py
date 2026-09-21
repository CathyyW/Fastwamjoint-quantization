"""Live, per-denoise-step CUDA Graphs (not fixed-observation replay).

Only the deterministic joint DiT core is captured. Noise generation, encoders,
the scheduler and the environment stay outside. Model weights/kernel settings
must not be changed after installation. One fixed layout per step is supported;
unexpected layouts or schedules fail rather than silently using stale inputs.
"""
from __future__ import annotations

from functools import wraps
import inspect
import logging
from typing import Any

import torch


def _signature(value: Any):
    if isinstance(value, torch.Tensor):
        if value.device.type != "cuda":
            raise ValueError("Live DiT Graph inputs must be CUDA tensors.")
        return ("tensor", tuple(value.shape), value.dtype, value.device)
    if isinstance(value, dict):
        return ("dict", tuple((key, _signature(item)) for key, item in value.items()))
    if isinstance(value, (tuple, list)):
        return (type(value).__name__, tuple(_signature(item) for item in value))
    if value is None or isinstance(value, (bool, int, float, str)):
        return (type(value).__name__, value)
    raise TypeError(f"Unsupported live Graph input: {type(value).__name__}")


def _clone(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _clone(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_clone(item) for item in value)
    return value


def _copy(target, source):
    if isinstance(target, torch.Tensor):
        target.copy_(source)
    elif isinstance(target, dict):
        for key in target:
            _copy(target[key], source[key])
    elif isinstance(target, (tuple, list)):
        for a, b in zip(target, source):
            _copy(a, b)


class LiveJointDiTGraph:
    def __init__(self, model, *, num_calls: int = 10, warmup: int = 2,
                 allow_numerical_mismatch: bool = False):
        if num_calls <= 0 or warmup < 1:
            raise ValueError("Graph calls and warmup must be positive.")
        self.model = model
        self.num_calls = num_calls
        self.warmup = warmup
        self.allow_numerical_mismatch = allow_numerical_mismatch
        self.validation_records = []
        self.entries = {}
        self.calls = 0
        self.chunks = 0
        self.replays = 0
        self._active = False
        self._installed = False

    def install(self):
        if self._installed or hasattr(self.model, "_rollout_graph"):
            raise RuntimeError("Live rollout Graph is already installed.")
        self.original_core = self.model._joint_denoise_core
        self.original_infer = self.model.infer_action
        self._old_core = self.model.__dict__.get("_joint_denoise_core")
        self._old_infer = self.model.__dict__.get("infer_action")
        signature = inspect.signature(self.original_infer)

        @wraps(self.original_infer)
        def infer(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            if int(bound.arguments.get("num_inference_steps", self.num_calls)) != self.num_calls:
                raise ValueError("Live Graph denoising schedule changed.")
            if float(bound.arguments.get("text_cfg_scale", 1.0)) != 1.0:
                raise ValueError("Live Graph currently requires text_cfg_scale=1.")
            if bound.arguments.get("compile_action_infer", False):
                raise ValueError("Use live DiT Graph without compile_action_infer.")
            if self._active:
                raise RuntimeError("Concurrent/reentrant Graph inference is not supported.")
            self.calls = 0
            self._active = True
            try:
                result = self.original_infer(*args, **kwargs)
                if self.calls != self.num_calls:
                    raise RuntimeError(f"Expected {self.num_calls} DiT calls, got {self.calls}.")
                self.chunks += 1
                if self.chunks == 1:
                    logging.warning("[live DiT Graph] ready: %d graphs; latest inputs copied per replay; peak %.2f GiB",
                                    len(self.entries), torch.cuda.max_memory_allocated() / 2**30)
                return result
            finally:
                self._active = False

        self.model._joint_denoise_core = self._call
        self.model.infer_action = infer
        self.model._rollout_graph = self
        self._installed = True
        return self

    def _call(self, *args, **kwargs):
        if not self._active or torch.is_grad_enabled():
            raise RuntimeError("Live DiT Graph is inference-only; use model.infer_action under no_grad.")
        index = self.calls
        if index >= self.num_calls:
            raise RuntimeError("Too many joint DiT calls in this action chunk.")
        tracker = getattr(self.model, "_wam_call_tracker", None)
        if tracker is not None and tracker.state.require() != index:
            raise RuntimeError("WAM call state does not match live Graph index.")
        values = (args, kwargs)
        signature = _signature(values)
        if index not in self.entries:
            # Independent pools avoid cross-graph output lifetime/ordering
            # assumptions. Returned tensors are cloned before any later replay.
            static_args, static_kwargs = _clone(values)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(self.warmup):
                    expected = self.original_core(*static_args, **static_kwargs)
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                outputs = self.original_core(*static_args, **static_kwargs)
            graph.replay()
            torch.cuda.synchronize()
            self._validate_outputs(outputs, expected, index)
            self.entries[index] = (signature, static_args, static_kwargs, graph, outputs)
            logging.warning("[live DiT Graph] captured step %d/%d (validation mode=%s)",
                            index + 1, self.num_calls,
                            "warn" if self.allow_numerical_mismatch else "strict")
        entry_signature, static_args, static_kwargs, graph, outputs = self.entries[index]
        if signature != entry_signature:
            raise ValueError("Live DiT Graph input layout/constants changed; refusing stale replay.")
        _copy((static_args, static_kwargs), values)
        graph.replay()
        if self.allow_numerical_mismatch:
            # Opt-in experimental runs still reject nonfinite outputs on every replay.
            for output in outputs:
                if not torch.isfinite(output).all():
                    raise RuntimeError("Nonfinite live Graph output during replay.")
        self.replays += 1
        self.calls += 1
        return _clone(outputs)

    def _validate_outputs(self, outputs, expected, index):
        if len(outputs) != len(expected):
            raise RuntimeError("Live Graph output count changed.")
        for output_index, (actual, reference) in enumerate(zip(outputs, expected)):
            if actual.shape != reference.shape or actual.dtype != reference.dtype or actual.device != reference.device:
                raise RuntimeError("Live Graph output layout changed.")
            if not torch.isfinite(actual).all() or not torch.isfinite(reference).all():
                raise RuntimeError("Nonfinite live Graph output during capture validation.")
            difference = actual.float() - reference.float()
            record = {"step": index, "output": output_index,
                      "max_abs": difference.abs().max().item(),
                      "rmse": difference.square().mean().sqrt().item(), "passed": True}
            try:
                torch.testing.assert_close(actual, reference, rtol=1e-3, atol=1e-3)
            except AssertionError:
                record["passed"] = False
                self.validation_records.append(record)
                if not self.allow_numerical_mismatch:
                    raise
                logging.warning("[live DiT Graph] numerical mismatch ACCEPTED by explicit opt-in: %s; "
                                "eager equivalence and SR equivalence are NOT established", record)
            else:
                self.validation_records.append(record)

    def remove(self):
        if self._active:
            raise RuntimeError("Cannot remove Graph during inference.")
        if self._installed:
            torch.cuda.synchronize()
            for name, previous in (("_joint_denoise_core", self._old_core), ("infer_action", self._old_infer)):
                if previous is None:
                    delattr(self.model, name)
                else:
                    setattr(self.model, name, previous)
            delattr(self.model, "_rollout_graph")
            self.entries.clear()
            self._installed = False
