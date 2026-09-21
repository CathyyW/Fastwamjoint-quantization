from __future__ import annotations

import torch
import torch.nn as nn

from fastwam_steerquant import (
    ActivationCache,
    ActivationCollector,
    GammaCalibrationConfig,
    DCalibrationConfig,
    DenoiseCallState,
    FastWAMStreamConfig,
    LazyActionSensitivityCollector,
    SensitivityAccumulator,
    SensitivityField,
    apply_checkpoint,
    calibrate_model,
    enumerate_fastwamjoint_linears,
    estimate_action_sensitivity,
    resolve_stream_layout,
)
from fastwam_steerquant.cache import CacheKey
from fastwam_steerquant.sensitivity import SiteSensitivity


class Attention(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.q = nn.Linear(hidden, hidden)
        self.k = nn.Linear(hidden, hidden)
        self.v = nn.Linear(hidden, hidden)
        self.o = nn.Linear(hidden, hidden)


class Block(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.self_attn = Attention(hidden)
        self.cross_attn = Attention(hidden)
        self.ffn = nn.Sequential(nn.Linear(hidden, hidden * 2), nn.GELU(), nn.Linear(hidden * 2, hidden))


class Expert(nn.Module):
    def __init__(self, hidden: int, layers: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(Block(hidden) for _ in range(layers))


class TinyFastWAM(nn.Module):
    def __init__(self, hidden: int = 4, layers: int = 1) -> None:
        super().__init__()
        self.video_expert = Expert(hidden, layers)
        self.action_expert = Expert(hidden, layers)


def test_topology_and_architecture_native_streams() -> None:
    model = TinyFastWAM(layers=2)
    sites = enumerate_fastwamjoint_linears(model)
    assert len(sites) == 40
    config = FastWAMStreamConfig(video_latent_frames=3, action_chunks=2)
    video = next(site for site in sites if site.expert == "video" and site.operation == "self_attn.q")
    action = next(site for site in sites if site.expert == "action" and site.operation == "ffn.0")
    context = next(site for site in sites if site.expert == "video" and site.operation == "cross_attn.k")
    assert resolve_stream_layout(video, 12, config).token_counts == (4, 4, 4)
    assert resolve_stream_layout(action, 8, config).token_counts == (4, 4)
    assert resolve_stream_layout(context, 6, config).token_counts == (5, 1)


def test_activation_collection_uses_call_and_stream_keys(tmp_path) -> None:
    model = TinyFastWAM()
    site = enumerate_fastwamjoint_linears(model)[0]
    state = DenoiseCallState(2)
    config = FastWAMStreamConfig(video_latent_frames=2, action_chunks=2)
    cache = ActivationCache(num_calls=2, max_rows_per_cell=32, seed=3)
    with ActivationCollector([site], state=state, stream_config=config, cache=cache):
        for call in range(2):
            state.set(call)
            site.module(torch.randn(1, 6, 4))
    assert len(cache.keys) == 4
    assert cache.values(CacheKey(site.index, 0, 0)).shape == (3, 4)
    loaded = ActivationCache.load(cache.save(tmp_path / "cache.pt"))
    assert loaded.keys == cache.keys


def test_rademacher_action_sensitivity() -> None:
    torch.manual_seed(4)
    model = TinyFastWAM()
    site = enumerate_fastwamjoint_linears(model)[0]
    state = DenoiseCallState(1)
    x = torch.randn(1, 6, 4)

    def run() -> torch.Tensor:
        state.set(0)
        return site.module(x)

    result = estimate_action_sensitivity(
        run,
        [site],
        state=state,
        stream_config=FastWAMStreamConfig(video_latent_frames=2, action_chunks=2),
        action_scale=1.0,
        weight_bits=4,
        activation_bits=4,
        num_probes=3,
        seed=5,
    )
    names, values = result[site.index]
    assert names == ("latent_frame_0", "latent_frame_1")
    assert values.shape == (1, 2)
    assert torch.isfinite(values).all() and (values >= 0).all()


def test_lazy_all_site_sensitivity_and_field_merge() -> None:
    torch.manual_seed(6)
    model = TinyFastWAM()
    sites = enumerate_fastwamjoint_linears(model)[:2]
    state = DenoiseCallState(1)
    config = FastWAMStreamConfig(video_latent_frames=2, action_chunks=2)
    x = torch.randn(1, 6, 4, requires_grad=True)
    with LazyActionSensitivityCollector(
        sites,
        state=state,
        stream_config=config,
        weight_bits=4,
        activation_bits=8,
    ) as collector:
        collector.reset_observation()
        collector.begin_probe()
        state.set(0)
        output = sites[1].module(torch.relu(sites[0].module(x)))
        collector.validate_forward()
        output.sum().backward()
        observed = collector.finish_observation(1)
    accumulator = SensitivityAccumulator(num_calls=1)
    for site in sites:
        names, values = observed[site.index]
        accumulator.add(site, names, values)
    field = accumulator.finalize()
    merged = SensitivityField.merge([field, field])
    assert torch.equal(merged.sites[0].observation_counts, torch.full((1, 2), 2))
    assert torch.allclose(merged.sites[0].values, field.sites[0].values)


def test_activation_cache_merge() -> None:
    left = ActivationCache(num_calls=1, max_rows_per_cell=4, seed=1)
    right = ActivationCache(num_calls=1, max_rows_per_cell=4, seed=2)
    key = CacheKey(0, 0, 0)
    left.add(key, torch.ones(3, 2), stream_name="stream")
    right.add(key, torch.full((3, 2), 2.0), stream_name="stream")
    merged = ActivationCache.merge([left, right])
    assert merged.values(key).shape == (4, 2)
    assert merged.seen_rows(key) == 6
    assert torch.equal(merged.channel_absmax(key), torch.full((2,), 2.0))


def test_calibration_checkpoint_and_runtime(tmp_path) -> None:
    torch.manual_seed(7)
    model = TinyFastWAM()
    site = enumerate_fastwamjoint_linears(model)[0]
    cache = ActivationCache(num_calls=2, max_rows_per_cell=64)
    stream_names = ("latent_frame_0", "latent_frame_1")
    for call in range(2):
        for stream, name in enumerate(stream_names):
            values = torch.randn(24, 4) * (1 + call + stream)
            cache.add(CacheKey(site.index, call, stream), values, stream_name=name)
    field = SensitivityField(
        num_calls=2,
        sites=(
            SiteSensitivity(
                site_index=site.index,
                module_name=site.module_name,
                stream_names=stream_names,
                values=torch.tensor([[1.0, 0.2], [0.4, 1.5]]),
                observation_counts=torch.ones(2, 2, dtype=torch.long),
            ),
        ),
    )
    checkpoint = calibrate_model(
        [site],
        cache=cache,
        sensitivity_field=field,
        stream_config=FastWAMStreamConfig(video_latent_frames=2, action_chunks=2),
        d_config=DCalibrationConfig(weight_bits=4, activation_bits=4, epochs=2),
        gamma_config=GammaCalibrationConfig(activation_bits=4, epochs=2),
    )
    checkpoint.validate()
    entry = checkpoint.sites[0]
    conservation = (entry.token_fractions * entry.gamma_gains.log()).sum(dim=1)
    assert torch.allclose(conservation, torch.zeros_like(conservation), atol=1e-5)

    loaded = type(checkpoint).load(checkpoint.save(tmp_path / "wam.pt"))
    state, count = apply_checkpoint(model, loaded)
    assert count == 1
    state.set(1)
    output = model.video_expert.blocks[0].self_attn.q(torch.randn(1, 6, 4))
    assert output.shape == (1, 6, 4)
    assert torch.isfinite(output).all()
