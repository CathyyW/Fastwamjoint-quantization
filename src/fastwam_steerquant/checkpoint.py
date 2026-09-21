from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
from typing import Any, Iterable

import torch


FORMAT = "fastwam_steerquant_calibration_v1"
ROTATED_FORMAT = "fastwam_steerquant_calibration_rht_v1"


@dataclass(frozen=True)
class QuantizedSite:
    site_index: int
    module_name: str
    expert: str
    operation: str
    stream_names: tuple[str, ...]
    qweight: torch.Tensor
    weight_scales: torch.Tensor
    input_scale: torch.Tensor
    gamma_gains: torch.Tensor
    clipping_ranges: torch.Tensor
    token_fractions: torch.Tensor
    d_initial_loss: float
    d_final_loss: float
    gamma_initial_losses: tuple[float, ...]
    gamma_final_losses: tuple[float, ...]
    input_rotation_signs: torch.Tensor | None = None
    rotation: str = "none"

    def validate(self, num_calls: int) -> None:
        if self.qweight.ndim != 2 or self.qweight.dtype != torch.int8:
            raise ValueError("qweight must be a two-dimensional int8 tensor.")
        out_features, in_features = self.qweight.shape
        if self.rotation not in ("none", "rht"):
            raise ValueError("Unsupported WAM checkpoint rotation.")
        signs = self.input_rotation_signs
        if (signs is None) != (self.rotation == "none"):
            raise ValueError("WAM rotation metadata/signs do not match.")
        if signs is not None and (signs.shape != (in_features,) or not torch.all(signs.abs() == 1)):
            raise ValueError("WAM rotation signs must be finite +/-1 per input channel.")
        streams = len(self.stream_names)
        if streams == 0 or len(set(self.stream_names)) != streams:
            raise ValueError("stream_names must be non-empty and unique.")
        if self.weight_scales.shape != (out_features, 1):
            raise ValueError("weight_scales must be per output channel.")
        if self.input_scale.shape != (in_features,):
            raise ValueError("input_scale shape does not match qweight.")
        if self.gamma_gains.shape != (num_calls, streams):
            raise ValueError("gamma_gains has an invalid shape.")
        if self.clipping_ranges.shape != (num_calls,):
            raise ValueError("clipping_ranges has an invalid shape.")
        if self.token_fractions.shape != self.gamma_gains.shape:
            raise ValueError("token_fractions must match gamma_gains.")
        positive = (self.weight_scales, self.input_scale, self.gamma_gains, self.clipping_ranges)
        if any(not torch.isfinite(value).all() or (value <= 0).any() for value in positive):
            raise ValueError("Checkpoint scales, gains, and clips must be finite and positive.")
        fractions = self.token_fractions
        if not torch.isfinite(fractions).all() or (fractions < 0).any():
            raise ValueError("token_fractions must be finite and non-negative.")
        if not torch.allclose(fractions.sum(dim=1), torch.ones(num_calls), atol=1e-6, rtol=0):
            raise ValueError("token_fractions must sum to one per call.")
        conservation = (fractions * self.gamma_gains.log()).sum(dim=1)
        if not torch.allclose(conservation, torch.zeros_like(conservation), atol=1e-5, rtol=0):
            raise ValueError("gamma gains violate within-call weighted-log conservation.")


@dataclass(frozen=True)
class QuantizationCheckpoint:
    weight_bits: int
    activation_bits: int
    num_calls: int
    stream_config: dict[str, Any]
    d_config: dict[str, Any]
    gamma_config: dict[str, Any]
    sites: tuple[QuantizedSite, ...]

    def validate(self) -> None:
        if not 2 <= self.weight_bits <= 8 or not 2 <= self.activation_bits <= 8:
            raise ValueError("Checkpoint bit widths must be in [2, 8].")
        if self.num_calls <= 0 or not self.sites:
            raise ValueError("Checkpoint needs at least one call and site.")
        names = [site.module_name for site in self.sites]
        if len(names) != len(set(names)):
            raise ValueError("Checkpoint module names must be unique.")
        for site in self.sites:
            site.validate(self.num_calls)
        if len({site.rotation for site in self.sites}) != 1:
            raise ValueError("Cannot mix rotated and unrotated WAM sites.")

    def state_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "format": ROTATED_FORMAT if self.sites[0].rotation != "none" else FORMAT,
            "weight_bits": self.weight_bits,
            "activation_bits": self.activation_bits,
            "num_calls": self.num_calls,
            "stream_config": self.stream_config,
            "d_config": self.d_config,
            "gamma_config": self.gamma_config,
            "sites": [
                {
                    **asdict(site),
                    "qweight": site.qweight.cpu(),
                    "input_rotation_signs": (None if site.input_rotation_signs is None
                                             else site.input_rotation_signs.cpu()),
                    "weight_scales": site.weight_scales.cpu(),
                    "input_scale": site.input_scale.cpu(),
                    "gamma_gains": site.gamma_gains.cpu(),
                    "clipping_ranges": site.clipping_ranges.cpu(),
                    "token_fractions": site.token_fractions.cpu(),
                }
                for site in self.sites
            ],
        }

    @classmethod
    def merge(cls, checkpoints: Iterable["QuantizationCheckpoint"]) -> "QuantizationCheckpoint":
        selected = tuple(checkpoints)
        if not selected:
            raise ValueError("At least one quantization checkpoint is required.")
        first = selected[0]
        metadata = (
            first.weight_bits,
            first.activation_bits,
            first.num_calls,
            first.stream_config,
            first.d_config,
            first.gamma_config,
        )
        for checkpoint in selected:
            checkpoint.validate()
            other = (
                checkpoint.weight_bits,
                checkpoint.activation_bits,
                checkpoint.num_calls,
                checkpoint.stream_config,
                checkpoint.d_config,
                checkpoint.gamma_config,
            )
            if other != metadata:
                raise ValueError("Quantization checkpoint configurations do not match.")
        sites = tuple(sorted(
            (site for checkpoint in selected for site in checkpoint.sites),
            key=lambda item: item.site_index,
        ))
        if len({site.site_index for site in sites}) != len(sites):
            raise ValueError("Quantization checkpoint shards overlap.")
        result = cls(
            weight_bits=first.weight_bits,
            activation_bits=first.activation_bits,
            num_calls=first.num_calls,
            stream_config=first.stream_config,
            d_config=first.d_config,
            gamma_config=first.gamma_config,
            sites=sites,
        )
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
    def from_state_dict(cls, payload: dict[str, Any]) -> "QuantizationCheckpoint":
        if payload.get("format") not in (FORMAT, ROTATED_FORMAT):
            raise ValueError(f"Expected checkpoint format {FORMAT!r} or {ROTATED_FORMAT!r}.")
        sites = tuple(
            QuantizedSite(
                site_index=int(item["site_index"]),
                module_name=str(item["module_name"]),
                expert=str(item["expert"]),
                operation=str(item["operation"]),
                stream_names=tuple(item["stream_names"]),
                qweight=torch.as_tensor(item["qweight"], dtype=torch.int8).cpu(),
                weight_scales=torch.as_tensor(item["weight_scales"], dtype=torch.float32).cpu(),
                input_scale=torch.as_tensor(item["input_scale"], dtype=torch.float32).cpu(),
                gamma_gains=torch.as_tensor(item["gamma_gains"], dtype=torch.float32).cpu(),
                clipping_ranges=torch.as_tensor(item["clipping_ranges"], dtype=torch.float32).cpu(),
                token_fractions=torch.as_tensor(item["token_fractions"], dtype=torch.float32).cpu(),
                d_initial_loss=float(item["d_initial_loss"]),
                d_final_loss=float(item["d_final_loss"]),
                gamma_initial_losses=tuple(float(value) for value in item["gamma_initial_losses"]),
                gamma_final_losses=tuple(float(value) for value in item["gamma_final_losses"]),
                input_rotation_signs=(None if item.get("input_rotation_signs") is None
                                      else torch.as_tensor(item["input_rotation_signs"], dtype=torch.float32)),
                rotation=str(item.get("rotation", "none")),
            )
            for item in payload["sites"]
        )
        result = cls(
            weight_bits=int(payload["weight_bits"]),
            activation_bits=int(payload["activation_bits"]),
            num_calls=int(payload["num_calls"]),
            stream_config=dict(payload["stream_config"]),
            d_config=dict(payload["d_config"]),
            gamma_config=dict(payload["gamma_config"]),
            sites=sites,
        )
        result.validate()
        rotated = result.sites[0].rotation != "none"
        if rotated != (payload["format"] == ROTATED_FORMAT):
            raise ValueError("Checkpoint format and rotation metadata disagree.")
        return result

    @classmethod
    def load(cls, path: str | Path) -> "QuantizationCheckpoint":
        try:
            payload = torch.load(Path(path).expanduser(), map_location="cpu", weights_only=True)
        except TypeError:
            payload = torch.load(Path(path).expanduser(), map_location="cpu")
        return cls.from_state_dict(payload)
