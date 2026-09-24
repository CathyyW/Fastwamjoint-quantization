from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal

from .checkpoint import QuantizationCheckpoint, QuantizedSite
from .quant import symmetric_fake_quant
from .state import DenoiseCallState
from .streams import FastWAMStreamConfig, expand_stream_gains, resolve_stream_layout
from .topology import LinearSite, resolve_site_module, resolve_site_parent


class WAMQuantLinear(nn.Module):
    """Reference or native W4A8/W4A4 Linear with static D and local gamma."""

    @classmethod
    def from_packed(cls, spec: dict, tensors: dict, *, state: DenoiseCallState,
                    stream_config: FastWAMStreamConfig, activation_bits: int):
        """Build a native Linear directly from deployment buffers, with no FP weight."""
        self = cls.__new__(cls)
        nn.Module.__init__(self)
        if activation_bits not in (4, 8):
            raise ValueError("Deployment supports W4A4 or W4A8.")
        self.backend = f"cutlass_w4a{activation_bits}"
        if activation_bits == 4:
            from .kernels.w4a4 import wam_ops as native_ops
        else:
            from .kernels.w4a8 import ops as native_ops
        self._native_ops = native_ops
        self.state, self.stream_config = state, stream_config
        self.weight_bits, self.activation_bits = 4, activation_bits
        self.rotation = spec["rotation"]
        if self.rotation not in ("none", "rht") or (self.rotation != "none" and activation_bits != 4):
            raise ValueError("RHT deployment requires W4A4.")
        self.stream_names = tuple(spec["stream_names"])
        self.supports_fused_adaln = False
        self.supports_fused_gate_residual = False
        for name in ("weight", "weight_scales", "input_scale", "gamma_gains",
                     "clipping_ranges", "bias", "input_rotation_signs"):
            self.register_buffer(name, tensors.get(name))
        self.site = LinearSite(spec["site_index"], spec["expert"], -1, spec["operation"],
                               spec["module_name"], None,
                               "context" if spec["operation"] in {"cross_attn.k", "cross_attn.v"}
                               else "expert_tokens")
        return self

    def __init__(
        self,
        source: nn.Linear,
        entry: QuantizedSite,
        *,
        state: DenoiseCallState,
        stream_config: FastWAMStreamConfig,
        weight_bits: int,
        activation_bits: int,
        backend: Literal["reference", "cutlass_w4a8", "cutlass_w4a4"] = "reference",
        fuse_block: bool = False,
    ) -> None:
        super().__init__()
        entry.validate(state.num_calls)
        # The reference backend needs the dequantized tensor for F.linear.  Build
        # it once when loading the checkpoint instead of repeating the identical
        # INT8-to-model-dtype conversion at every denoising call.
        if backend not in ("reference", "cutlass_w4a8", "cutlass_w4a4"):
            raise ValueError(f"Unsupported WAM backend: {backend}.")
        native = backend != "reference"
        if native and (weight_bits != 4 or activation_bits != (4 if backend == "cutlass_w4a4" else 8)):
            raise ValueError(f"WAM {backend} does not match W{weight_bits}A{activation_bits}.")
        device = source.weight.device
        self.rotation = entry.rotation
        self.register_buffer("input_rotation_signs", None if entry.input_rotation_signs is None
                             else entry.input_rotation_signs.float().to(device=device))
        if self.rotation != "none" and backend not in ("reference", "cutlass_w4a4"):
            raise ValueError("Rotated WAM requires reference or cutlass_w4a4.")
        if self.rotation != "none" and fuse_block:
            raise ValueError("Rotated FastWAM block fusion is not validated; leave fuse_block=False.")
        if native:
            from .kernels.w4a8 import pack_signed_int4

            if backend == "cutlass_w4a4":
                from .kernels.w4a4 import wam_ops as native_ops
            else:
                from .kernels.w4a8 import ops as native_ops
            self._native_ops = native_ops

            self.register_buffer("weight", pack_signed_int4(entry.qweight).to(device=device))
            self.register_buffer("weight_scales", entry.weight_scales.float().reshape(-1).to(device=device))
            self.register_buffer("input_scale", entry.input_scale.float().reciprocal().to(device=device))
            self.register_buffer("gamma_gains", entry.gamma_gains.float().to(device=device))
            self.register_buffer("clipping_ranges", (entry.clipping_ranges.float() / (2 ** (activation_bits - 1) - 1)).to(device=device))
        else:
            self.register_buffer(
                "weight",
                entry.qweight.to(device=device, dtype=source.weight.dtype)
                * entry.weight_scales.to(device=device, dtype=source.weight.dtype),
            )
            self.register_buffer("weight_scales", None)
            self.register_buffer("input_scale", entry.input_scale.clone())
            self.register_buffer("gamma_gains", entry.gamma_gains.clone())
            self.register_buffer("clipping_ranges", entry.clipping_ranges.clone())
        self.register_buffer("bias", None if source.bias is None else source.bias.detach().clone())
        self.state = state
        self.stream_config = stream_config
        self.weight_bits = int(weight_bits)
        self.activation_bits = int(activation_bits)
        self.backend = backend
        self.stream_names = entry.stream_names
        self.supports_fused_adaln = bool(
            fuse_block and native and entry.operation == "ffn.0"
        )
        self.supports_fused_gate_residual = bool(
            fuse_block and native
            and entry.operation in {"self_attn.o", "ffn.2"}
        )
        self.site = LinearSite(
            index=entry.site_index,
            expert=entry.expert,  # type: ignore[arg-type]
            block_index=-1,
            operation=entry.operation,
            module_name=entry.module_name,
            # Runtime stream dispatch only consumes expert/operation/kind.
            # Retaining ``source`` here would keep every replaced BF16 Linear
            # on the GPU and invalidate packed-model peak-memory measurements.
            module=None,
            stream_kind="context" if entry.operation in {"cross_attn.k", "cross_attn.v"} else "expert_tokens",
        )

    @property
    def in_features(self) -> int:
        return self.weight.shape[1] * (2 if self.backend != "reference" else 1)

    @property
    def out_features(self) -> int:
        return self.weight.shape[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        call = self.state.require()
        layout = resolve_stream_layout(self.site, x.shape[-2], self.stream_config)
        if self.backend != "reference":
            if layout.names != self.stream_names:
                raise RuntimeError(f"WAM checkpoint stream names differ from {self.site.module_name}: {layout.names}.")
            assert self.weight_scales is not None
            if self.input_rotation_signs is not None:
                return self._native_ops.symmetric_scheduled_stream_rht_linear(
                    x, self.weight, self.weight_scales, self.bias, self.input_scale,
                    self.clipping_ranges, self.gamma_gains, layout.token_counts, call,
                    self.input_rotation_signs,
                )
            return self._native_ops.symmetric_scheduled_stream_linear(
                x, self.weight, self.weight_scales, self.bias, self.input_scale,
                self.clipping_ranges, self.gamma_gains, layout.token_counts, call,
            )
        if self.input_rotation_signs is not None:
            from .rht import rht_transform
            x = rht_transform(x.float(), self.input_rotation_signs, self.rotation).to(x.dtype)
        gains = self.gamma_gains[call].to(device=x.device, dtype=torch.float32)
        if gains.numel() != layout.num_streams:
            raise RuntimeError(
                f"Runtime stream count changed for {self.site.module_name}: "
                f"checkpoint={gains.numel()}, input={layout.num_streams}."
            )
        x_prime = x.float() / self.input_scale.to(x.device)[None, :]
        expanded = expand_stream_gains(x_prime, gains, layout)
        clip = self.clipping_ranges[call].to(x.device)
        scale = clip / (2 ** (self.activation_bits - 1) - 1)
        qx = (symmetric_fake_quant(
            x_prime * expanded,
            bits=self.activation_bits,
            scale=scale,
            ste=False,
        ) / expanded).to(dtype=x.dtype)
        bias = None if self.bias is None else self.bias.to(device=x.device, dtype=x.dtype)
        return F.linear(qx, self.weight, bias)

    def forward_adaln(
        self, x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor, epsilon: float,
    ) -> torch.Tensor:
        if not self.supports_fused_adaln:
            raise RuntimeError("WAM AdaLN block fusion was not enabled for this site.")
        layout = resolve_stream_layout(self.site, x.shape[-2], self.stream_config)
        if layout.names != self.stream_names:
            raise RuntimeError("WAM fused AdaLN stream layout changed from the checkpoint.")
        assert self.weight_scales is not None
        return self._native_ops.symmetric_scheduled_stream_adaln_linear(
            x, self.weight, self.weight_scales, self.bias, self.input_scale,
            self.clipping_ranges, self.gamma_gains, layout.token_counts, self.state.require(),
            scale, shift, epsilon,
        )

    def forward_gate_residual(
        self, x: torch.Tensor, residual: torch.Tensor, gate: torch.Tensor,
    ) -> torch.Tensor:
        if not self.supports_fused_gate_residual:
            raise RuntimeError("WAM gate/residual block fusion was not enabled for this site.")
        layout = resolve_stream_layout(self.site, x.shape[-2], self.stream_config)
        if layout.names != self.stream_names:
            raise RuntimeError("WAM fused residual stream layout changed from the checkpoint.")
        assert self.weight_scales is not None
        return self._native_ops.symmetric_scheduled_stream_gate_residual_linear(
            x, self.weight, self.weight_scales, self.bias, self.input_scale,
            self.clipping_ranges, self.gamma_gains, layout.token_counts, self.state.require(),
            residual, gate,
        )


def set_wam_block_fusions(model: nn.Module, enabled: bool) -> int:
    """Toggle only the 3 supported WAM operations per FastWAM expert block."""
    if getattr(model, "_rollout_graph", None) is not None:
        raise RuntimeError("Cannot change block fusion after CUDA Graph installation.")
    if enabled and any(isinstance(m, WAMQuantLinear) and m.rotation != "none" for m in model.modules()):
        raise ValueError("Rotated FastWAM block fusion is not validated.")
    count = 0
    for module in model.modules():
        if not isinstance(module, WAMQuantLinear) or module.backend == "reference":
            continue
        module.supports_fused_adaln = bool(enabled and module.site.operation == "ffn.0")
        module.supports_fused_gate_residual = bool(
            enabled and module.site.operation in {"self_attn.o", "ffn.2"}
        )
        count += int(module.supports_fused_adaln or module.supports_fused_gate_residual)
    return count


def apply_checkpoint(
    model: nn.Module,
    checkpoint: QuantizationCheckpoint,
    *,
    state: DenoiseCallState | None = None,
    backend: Literal["reference", "cutlass_w4a8", "cutlass_w4a4"] = "reference",
    fuse_block: bool = False,
) -> tuple[DenoiseCallState, int]:
    checkpoint.validate()
    if any(e.rotation != "none" for e in checkpoint.sites) and (fuse_block or backend == "cutlass_w4a8"):
        raise ValueError("Rotated WAM requires W4A4/reference with block fusion disabled.")
    if backend not in ("reference", "cutlass_w4a8", "cutlass_w4a4"):
        raise ValueError(f"Unsupported WAM backend: {backend}.")
    if backend != "reference" and (checkpoint.weight_bits != 4 or backend != f"cutlass_w4a{checkpoint.activation_bits}"):
        raise ValueError("WAM native backend and checkpoint bit widths must match.")
    state = DenoiseCallState(checkpoint.num_calls) if state is None else state
    if state.num_calls != checkpoint.num_calls:
        raise ValueError("Runtime call state and checkpoint call counts differ.")
    stream_config = FastWAMStreamConfig(**checkpoint.stream_config)
    # Validate topology before modifying the model; then replace one site at a
    # time so the previous FP Linear and the native packed weight never require
    # a second 600-layer copy on the GPU.
    for entry in checkpoint.sites:
        module = resolve_site_module(model, entry.module_name)
        if not isinstance(module, nn.Linear):
            raise TypeError(f"{entry.module_name} is not an nn.Linear.")
        if module.weight.shape != entry.qweight.shape:
            raise ValueError(f"WAM checkpoint shape differs from {entry.module_name}.")
        if backend != "reference":
            alignment = 256 if backend == "cutlass_w4a4" else 32
            if module.in_features % alignment or module.out_features % 8:
                raise ValueError(f"Unsupported {backend} shape at {entry.module_name}.")
    replaced = 0
    for entry in checkpoint.sites:
        module = resolve_site_module(model, entry.module_name)
        assert isinstance(module, nn.Linear)
        parent, child_name = resolve_site_parent(model, entry.module_name)
        replacement = WAMQuantLinear(
            module,
            entry,
            state=state,
            stream_config=stream_config,
            weight_bits=checkpoint.weight_bits,
            activation_bits=checkpoint.activation_bits,
            backend=backend,
            fuse_block=fuse_block,
        ).to(module.weight.device)
        replacement.train(module.training)
        setattr(parent, child_name, replacement)
        replaced += 1
    return state, replaced
