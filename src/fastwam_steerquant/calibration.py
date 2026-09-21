from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Callable, Iterable

import torch
import torch.nn as nn

from .cache import ActivationCache, CacheKey
from .checkpoint import QuantizationCheckpoint, QuantizedSite
from .quant import quantize_weight_per_channel, symmetric_fake_quant
from .sensitivity import SensitivityField, SiteSensitivity
from .streams import FastWAMStreamConfig
from .topology import LinearSite


@dataclass(frozen=True)
class DCalibrationConfig:
    weight_bits: int = 4
    activation_bits: int = 8
    epochs: int = 20
    learning_rate: float = 2e-2
    batch_size: int = 128
    smoothquant_alpha: float = 0.5
    log_scale_bound: float = 6.0

    def validate(self) -> None:
        if not 2 <= self.weight_bits <= 8 or not 2 <= self.activation_bits <= 8:
            raise ValueError("D calibration bit widths must be in [2, 8].")
        if self.epochs < 0 or self.learning_rate <= 0 or self.batch_size <= 0:
            raise ValueError("Invalid D optimization schedule.")
        if not 0 <= self.smoothquant_alpha <= 1 or self.log_scale_bound <= 0:
            raise ValueError("Invalid D scale configuration.")


@dataclass(frozen=True)
class GammaCalibrationConfig:
    activation_bits: int = 8
    epochs: int = 30
    gain_learning_rate: float = 2e-2
    clip_learning_rate: float = 1e-2
    batch_size: int = 128
    gamma_min: float = 0.25
    gamma_max: float = 4.0
    clip_min_ratio: float = 0.05
    clip_max_ratio: float = 2.0

    def validate(self) -> None:
        if not 2 <= self.activation_bits <= 8 or self.epochs < 0 or self.batch_size <= 0:
            raise ValueError("Invalid gamma calibration bit width or schedule.")
        if self.gain_learning_rate <= 0 or self.clip_learning_rate <= 0:
            raise ValueError("Gamma and clip learning rates must be positive.")
        if not 0 < self.gamma_min < 1 < self.gamma_max:
            raise ValueError("gamma bounds must contain one.")
        if not 0 < self.clip_min_ratio <= 1 <= self.clip_max_ratio:
            raise ValueError("clip ratios must contain one.")


@dataclass(frozen=True)
class DCalibrationResult:
    input_scale: torch.Tensor
    qweight: torch.Tensor
    weight_scales: torch.Tensor
    initial_loss: float
    final_loss: float


@dataclass(frozen=True)
class GammaCalibrationResult:
    gains: torch.Tensor
    clipping_ranges: torch.Tensor
    token_fractions: torch.Tensor
    initial_losses: tuple[float, ...]
    final_losses: tuple[float, ...]


def _importance(
    sensitivity: SiteSensitivity,
    *,
    rho: float,
    eps: float,
    timestep_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    energy = (sensitivity.values.float().square() + float(eps)).pow(float(rho))
    gamma = energy / energy.sum(dim=1, keepdim=True).clamp_min(1e-12)
    d = energy * timestep_weights[:, None]
    d = d / d.sum().clamp_min(1e-12)
    return d, gamma


def _project_log_scale_(value: torch.Tensor, bound: float) -> None:
    with torch.no_grad():
        value.sub_(value.mean()).clamp_(-bound, bound)


def _site_cells(cache: ActivationCache, site: LinearSite, sensitivity: SiteSensitivity):
    cells = []
    for call in range(cache.num_calls):
        for stream, stream_name in enumerate(sensitivity.stream_names):
            key = CacheKey(site.index, call, stream)
            if key not in cache.keys:
                raise ValueError(f"Missing activation cache cell {key}.")
            if cache.stream_name(key) != stream_name:
                raise ValueError(f"Cache/sensitivity stream mismatch for {key}.")
            cells.append((key, cache.values(key)))
    return cells


def calibrate_site_d(
    site: LinearSite,
    *,
    cache: ActivationCache,
    sensitivity: SiteSensitivity,
    d_weights: torch.Tensor,
    config: DCalibrationConfig,
) -> DCalibrationResult:
    config.validate()
    cells = _site_cells(cache, site, sensitivity)
    weight = site.module.weight.detach().float()
    activation_salience = torch.zeros(weight.shape[1])
    call_channel_absmax = torch.zeros(cache.num_calls, weight.shape[1])
    for key, _values in cells:
        maximum = cache.channel_absmax(key)
        activation_salience.add_(maximum, alpha=float(d_weights[key.call_index, key.stream_index]))
        call_channel_absmax[key.call_index] = torch.maximum(call_channel_absmax[key.call_index], maximum)
    weight_salience = weight.abs().amax(dim=0).cpu().clamp_min(1e-8)
    initial = activation_salience.clamp_min(1e-8).pow(config.smoothquant_alpha)
    initial /= weight_salience.pow(1 - config.smoothquant_alpha)
    log_d = nn.Parameter(initial.clamp_min(1e-8).log().to(weight.device))
    _project_log_scale_(log_d, config.log_scale_bound)
    optimizer = torch.optim.Adam([log_d], lr=config.learning_rate)

    def loss_for(d: torch.Tensor, *, ste: bool) -> torch.Tensor:
        transformed_weight = weight.to(d.device) * d[None, :]
        w_scale = transformed_weight.detach().abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
        w_scale /= 2 ** (config.weight_bits - 1) - 1
        qw = symmetric_fake_quant(transformed_weight, bits=config.weight_bits, scale=w_scale, ste=ste)
        transformed_max = call_channel_absmax.to(d.device) / d[None, :]
        a_scale = transformed_max.amax(dim=1).clamp_min(1e-8)
        a_scale /= 2 ** (config.activation_bits - 1) - 1
        total = d.new_zeros(())
        for key, values in cells:
            x = values.to(d.device)
            count = x.shape[0]
            for start in range(0, count, config.batch_size):
                chunk = x[start : start + config.batch_size]
                qx = symmetric_fake_quant(
                    chunk / d,
                    bits=config.activation_bits,
                    scale=a_scale[key.call_index],
                    ste=ste,
                )
                residual = qx @ qw.T - chunk @ weight.to(d.device).T
                total = total + residual.square().sum() * (
                    d_weights[key.call_index, key.stream_index].to(d.device) / count
                )
        return total

    with torch.no_grad():
        initial_loss = float(loss_for(log_d.exp(), ste=False).cpu())
    best_loss = initial_loss
    best_d = log_d.detach().exp().clone()
    for _epoch in range(config.epochs):
        optimizer.zero_grad(set_to_none=True)
        for key, values in cells:
            x = values.to(weight.device)
            count = x.shape[0]
            for start in range(0, count, config.batch_size):
                d = log_d.exp()
                transformed_weight = weight * d[None, :]
                weight_scale = transformed_weight.detach().abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
                weight_scale /= 2 ** (config.weight_bits - 1) - 1
                quantized_weight = symmetric_fake_quant(
                    transformed_weight,
                    bits=config.weight_bits,
                    scale=weight_scale,
                    ste=True,
                )
                transformed_max = call_channel_absmax.to(d.device) / d[None, :]
                activation_scale = transformed_max.amax(dim=1).clamp_min(1e-8)
                activation_scale /= 2 ** (config.activation_bits - 1) - 1
                chunk = x[start : start + config.batch_size]
                quantized_x = symmetric_fake_quant(
                    chunk / d,
                    bits=config.activation_bits,
                    scale=activation_scale[key.call_index],
                    ste=True,
                )
                residual = quantized_x @ quantized_weight.T - chunk @ weight.T
                chunk_loss = residual.square().sum() * (
                    d_weights[key.call_index, key.stream_index].to(d.device) / count
                )
                chunk_loss.backward()
        optimizer.step()
        _project_log_scale_(log_d, config.log_scale_bound)
        with torch.no_grad():
            hard = float(loss_for(log_d.exp(), ste=False).cpu())
        if hard < best_loss:
            best_loss = hard
            best_d = log_d.detach().exp().clone()
    qweight, scales = quantize_weight_per_channel(weight.to(best_d.device) * best_d[None, :], bits=config.weight_bits)
    return DCalibrationResult(best_d.float(), qweight.cpu(), scales.cpu(), initial_loss, best_loss)


def _constrained_gains(
    raw: torch.Tensor,
    fractions: torch.Tensor,
    *,
    lower: float,
    upper: float,
) -> torch.Tensor:
    centered = raw - (raw * fractions).sum()
    positive = centered.max().clamp_min(torch.finfo(centered.dtype).eps)
    negative = (-centered.min()).clamp_min(torch.finfo(centered.dtype).eps)
    shrink = torch.minimum(
        centered.new_ones(()),
        torch.minimum(centered.new_tensor(math.log(upper)) / positive, centered.new_tensor(-math.log(lower)) / negative),
    )
    return (centered * shrink).exp()


def calibrate_site_gamma(
    site: LinearSite,
    *,
    cache: ActivationCache,
    sensitivity: SiteSensitivity,
    gamma_weights: torch.Tensor,
    d_result: DCalibrationResult,
    config: GammaCalibrationConfig,
) -> GammaCalibrationResult:
    config.validate()
    cells = _site_cells(cache, site, sensitivity)
    device = site.module.weight.device
    weight = site.module.weight.detach().float().to(device)
    d = d_result.input_scale.to(device)
    qw = d_result.qweight.to(device).float() * d_result.weight_scales.to(device)
    num_streams = len(sensitivity.stream_names)
    all_gains = []
    all_clips = []
    all_fractions = []
    initial_losses = []
    final_losses = []

    for call in range(cache.num_calls):
        call_cells = [(key, values) for key, values in cells if key.call_index == call]
        counts = torch.tensor([cache.seen_rows(key) for key, _ in call_cells], device=device, dtype=torch.float32)
        fractions = counts / counts.sum()
        initial_clip = max(float((cache.channel_absmax(key).to(device) / d).amax()) for key, _ in call_cells)
        raw_gains = nn.Parameter(torch.zeros(num_streams, device=device))
        raw_clip = nn.Parameter(torch.tensor(math.log(initial_clip), device=device))
        optimizer = torch.optim.Adam(
            [
                {"params": [raw_gains], "lr": config.gain_learning_rate},
                {"params": [raw_clip], "lr": config.clip_learning_rate},
            ]
        )
        min_log_clip = math.log(initial_clip * config.clip_min_ratio)
        max_log_clip = math.log(initial_clip * config.clip_max_ratio)

        def current() -> tuple[torch.Tensor, torch.Tensor]:
            gains = _constrained_gains(
                raw_gains,
                fractions,
                lower=config.gamma_min,
                upper=config.gamma_max,
            )
            return gains, raw_clip.clamp(min_log_clip, max_log_clip).exp()

        def loss_for(gains: torch.Tensor, clip: torch.Tensor, *, ste: bool) -> torch.Tensor:
            total = gains.new_zeros(())
            scale = clip / (2 ** (config.activation_bits - 1) - 1)
            for key, values in call_cells:
                stream = key.stream_index
                x = values.to(device)
                count = x.shape[0]
                for start in range(0, count, config.batch_size):
                    chunk = x[start : start + config.batch_size]
                    x_prime = chunk / d
                    qx = symmetric_fake_quant(
                        gains[stream] * x_prime,
                        bits=config.activation_bits,
                        scale=scale,
                        ste=ste,
                    ) / gains[stream]
                    residual = qx @ qw.T - chunk @ weight.T
                    total = total + residual.square().sum() * (
                        gamma_weights[call, stream].to(device) / count
                    )
            return total

        with torch.no_grad():
            gains, clip = current()
            initial_loss = float(loss_for(gains, clip, ste=False).cpu())
        best_loss = initial_loss
        best_gains, best_clip = gains.detach().clone(), clip.detach().clone()
        for _epoch in range(config.epochs):
            optimizer.zero_grad(set_to_none=True)
            for key, values in call_cells:
                stream = key.stream_index
                x = values.to(device)
                count = x.shape[0]
                for start in range(0, count, config.batch_size):
                    gains, clip = current()
                    scale = clip / (2 ** (config.activation_bits - 1) - 1)
                    chunk = x[start : start + config.batch_size]
                    x_prime = chunk / d
                    quantized_x = symmetric_fake_quant(
                        gains[stream] * x_prime,
                        bits=config.activation_bits,
                        scale=scale,
                        ste=True,
                    ) / gains[stream]
                    residual = quantized_x @ qw.T - chunk @ weight.T
                    chunk_loss = residual.square().sum() * (
                        gamma_weights[call, stream].to(device) / count
                    )
                    chunk_loss.backward()
            optimizer.step()
            with torch.no_grad():
                raw_clip.clamp_(min_log_clip, max_log_clip)
                gains, clip = current()
                hard = float(loss_for(gains, clip, ste=False).cpu())
            if hard < best_loss:
                best_loss = hard
                best_gains, best_clip = gains.detach().clone(), clip.detach().clone()
        all_gains.append(best_gains.cpu())
        all_clips.append(best_clip.cpu())
        all_fractions.append(fractions.cpu())
        initial_losses.append(initial_loss)
        final_losses.append(best_loss)
    return GammaCalibrationResult(
        gains=torch.stack(all_gains),
        clipping_ranges=torch.stack(all_clips),
        token_fractions=torch.stack(all_fractions),
        initial_losses=tuple(initial_losses),
        final_losses=tuple(final_losses),
    )


def calibrate_model(
    sites: Iterable[LinearSite],
    *,
    cache: ActivationCache,
    sensitivity_field: SensitivityField,
    stream_config: FastWAMStreamConfig,
    d_config: DCalibrationConfig | None = None,
    gamma_config: GammaCalibrationConfig | None = None,
    timestep_weights: torch.Tensor | None = None,
    rho: float = 1.0,
    eps: float = 1e-8,
    progress: Callable[[int, int, str], None] | None = None,
    resume_sites: dict[int, QuantizedSite] | None = None,
    on_site_complete: Callable[[QuantizedSite], None] | None = None,
) -> QuantizationCheckpoint:
    selected = tuple(sites)
    expected_rotation = cache.rotation_config.get("rotation", "none")
    if sensitivity_field.rotation_config != cache.rotation_config:
        raise ValueError("WAM sensitivity and activation cache rotation domains/seeds do not match.")
    if any(getattr(site.module, "_wam_rotation", "none") != expected_rotation for site in selected):
        raise ValueError("WAM model and activation cache rotation domains do not match.")
    if any(getattr(site.module, "_wam_rotation_config", {}) != cache.rotation_config for site in selected):
        raise ValueError("WAM model and activation cache rotation seeds/contracts do not match.")
    sensitivity_field.validate()
    d_config = DCalibrationConfig() if d_config is None else d_config
    gamma_config = GammaCalibrationConfig(activation_bits=d_config.activation_bits) if gamma_config is None else gamma_config
    d_config.validate()
    gamma_config.validate()
    if d_config.activation_bits != gamma_config.activation_bits:
        raise ValueError("D and gamma activation bit widths must match.")
    if cache.num_calls != sensitivity_field.num_calls:
        raise ValueError("Cache and sensitivity call counts differ.")
    if not 0 < rho <= 1 or eps <= 0:
        raise ValueError("rho must be in (0, 1] and eps must be positive.")
    if timestep_weights is None:
        timestep_weights = torch.ones(cache.num_calls) / cache.num_calls
    else:
        timestep_weights = torch.as_tensor(timestep_weights, dtype=torch.float32)
        timestep_weights = timestep_weights / timestep_weights.sum()

    entries = []
    for position, site in enumerate(selected):
        if progress is not None:
            progress(position, len(selected), site.module_name)
        if resume_sites is not None and site.index in resume_sites:
            entry = resume_sites[site.index]
            entry.validate(cache.num_calls)
            if entry.module_name != site.module_name or entry.qweight.shape != site.module.weight.shape:
                raise ValueError("Resumed calibration site does not match the model.")
            signs = getattr(site.module, "_wam_rotation_signs", None)
            if entry.rotation != getattr(site.module, "_wam_rotation", "none") or (
                signs is not None and not torch.equal(signs.cpu(), entry.input_rotation_signs.cpu())
            ):
                raise ValueError("Resumed calibration rotation does not match the model.")
            entries.append(entry)
            continue
        sensitivity = sensitivity_field.by_site(site.index)
        if sensitivity.module_name != site.module_name:
            raise ValueError("Topology and sensitivity module names differ.")
        d_weights, gamma_weights = _importance(
            sensitivity,
            rho=rho,
            eps=eps,
            timestep_weights=timestep_weights,
        )
        d_result = calibrate_site_d(
            site,
            cache=cache,
            sensitivity=sensitivity,
            d_weights=d_weights,
            config=d_config,
        )
        gamma_result = calibrate_site_gamma(
            site,
            cache=cache,
            sensitivity=sensitivity,
            gamma_weights=gamma_weights,
            d_result=d_result,
            config=gamma_config,
        )
        entries.append(
            QuantizedSite(
                site_index=site.index,
                module_name=site.module_name,
                expert=site.expert,
                operation=site.operation,
                stream_names=sensitivity.stream_names,
                qweight=d_result.qweight,
                weight_scales=d_result.weight_scales,
                input_scale=d_result.input_scale.cpu(),
                gamma_gains=gamma_result.gains,
                clipping_ranges=gamma_result.clipping_ranges,
                token_fractions=gamma_result.token_fractions,
                d_initial_loss=d_result.initial_loss,
                d_final_loss=d_result.final_loss,
                gamma_initial_losses=gamma_result.initial_losses,
                gamma_final_losses=gamma_result.final_losses,
                input_rotation_signs=(None if not hasattr(site.module, "_wam_rotation_signs")
                                      else site.module._wam_rotation_signs.detach().cpu()),
                rotation=getattr(site.module, "_wam_rotation", "none"),
            )
        )
        if on_site_complete is not None:
            on_site_complete(entries[-1])
    checkpoint = QuantizationCheckpoint(
        weight_bits=d_config.weight_bits,
        activation_bits=d_config.activation_bits,
        num_calls=cache.num_calls,
        stream_config=asdict(stream_config),
        d_config=asdict(d_config),
        gamma_config=asdict(gamma_config),
        sites=tuple(entries),
    )
    checkpoint.validate()
    return checkpoint
