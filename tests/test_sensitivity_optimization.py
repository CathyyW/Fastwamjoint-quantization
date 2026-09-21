import pytest
import torch

from fastwam_steerquant import (
    ActivationCache, ActivationCollector, DenoiseCallState, FastWAMStreamConfig,
    LazyActionSensitivityCollector, enumerate_fastwamjoint_linears,
)
from fastwam_steerquant.recovery import sensitivity_commit_due, atomic_snapshot
from fastwam_steerquant.cache import CacheKey
from test_pipeline import TinyFastWAM


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_device_stats_and_shared_cache_match_legacy_exactly(device, bits, dtype):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(712)
    model = TinyFastWAM(hidden=16).to(device=device, dtype=dtype).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    all_sites = enumerate_fastwamjoint_linears(model)
    sites = [s for s in all_sites if s.operation in ("self_attn.q", "cross_attn.k")]
    state = DenoiseCallState(3)
    config = FastWAMStreamConfig(video_latent_frames=3, action_chunks=2, proprio_tokens=1)
    inputs = [torch.randn(2, 6, 16, device=device, dtype=dtype) for _ in range(2)]
    probes = [torch.randn(2, 6, 16, device=device, dtype=dtype) for _ in range(4)]
    results = []
    caches = []
    for optimized in (False, True):
        cache = ActivationCache(num_calls=3, max_rows_per_cell=7, seed=42, preserve_source_dtype=True)
        observations = []
        with LazyActionSensitivityCollector(sites, state=state, stream_config=config,
                                            weight_bits=4, activation_bits=bits,
                                            accumulate_on_device=optimized,
                                            deduplicate_captures=optimized) as collector:
            for value in inputs:
                collector.reset_observation()
                for probe_index, probe in enumerate(probes):
                    collector.begin_probe()
                    separate = ActivationCollector(sites, state=state, stream_config=config, cache=cache)
                    if probe_index == 0:
                        if optimized:
                            collector.activation_cache = cache
                        else:
                            separate.install()
                    x = value.detach().clone().requires_grad_(True)
                    for call in range(3):
                        state.set(call)
                        for site in sites:
                            x = site.module(x).tanh()
                    collector.validate_forward()
                    separate.remove()
                    collector.activation_cache = None
                    (x * probe).sum().backward()
                observations.append(collector.finish_observation(4))
        results.append(observations)
        caches.append(cache)
    for a, b in zip(*results):
        for index in a:
            assert a[index][0] == b[index][0]
            torch.testing.assert_close(a[index][1], b[index][1], rtol=0, atol=0)
    for key in caches[0].keys:
        torch.testing.assert_close(caches[0].values(key), caches[1].values(key), rtol=0, atol=0)
        torch.testing.assert_close(caches[0].channel_absmax(key), caches[1].channel_absmax(key), rtol=0, atol=0)
        assert caches[0].seen_rows(key) == caches[1].seen_rows(key)
        torch.testing.assert_close(caches[0]._cells[key].priorities, caches[1]._cells[key].priorities,
                                   rtol=0, atol=0)
    assert torch.equal(caches[0]._generator.get_state(), caches[1]._generator.get_state())


def test_commit_interval_always_commits_at_recycle_or_end():
    assert not sensitivity_commit_due(1, 100, 1, 4, False)
    assert sensitivity_commit_due(2, 100, 2, 4, True)
    assert sensitivity_commit_due(4, 100, 4, 4, False)
    assert sensitivity_commit_due(100, 100, 1, 4, False)
    with pytest.raises(ValueError):
        sensitivity_commit_due(1, 100, 1, 0, False)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_input_offload_dedup_preserves_mutation_and_lifetime():
    collector = LazyActionSensitivityCollector([], state=DenoiseCallState(1),
        stream_config=FastWAMStreamConfig(), weight_bits=4, activation_bits=4,
        deduplicate_captures=True)
    x = torch.ones(2, 4, device="cuda")
    a = collector._offload_input(x)
    assert collector._offload_input(x) is a
    x.add_(1)
    b = collector._offload_input(x)
    assert b is not a
    assert torch.equal(a, torch.ones(2, 4))
    assert torch.equal(b, torch.full((2, 4), 2.))
    y = x.clone()
    assert collector._offload_input(y) is not b
    collector.begin_probe()
    assert not collector._input_offloads


def test_sparse_snapshot_recomputes_uncommitted_reservoir_exactly(tmp_path):
    values = [torch.randn(13, 8, generator=torch.Generator().manual_seed(i)) for i in range(5)]
    key = CacheKey(0, 0, 0)
    direct = ActivationCache(num_calls=1, max_rows_per_cell=9, seed=42)
    resumed = ActivationCache(num_calls=1, max_rows_per_cell=9, seed=42)
    for i, value in enumerate(values):
        direct.add(key, value, stream_name="s")
        if i < 3:
            resumed.add(key, value, stream_name="s")
        if i == 1:
            atomic_snapshot({"cache": resumed.state_dict(), "completed": 2}, tmp_path / "progress.pt")
    # Observation 3 was computed but never committed; discard it and replay.
    payload = torch.load(tmp_path / "progress.pt", weights_only=True)
    resumed = ActivationCache.from_state_dict(payload['cache'])
    for value in values[payload['completed']:]:
        resumed.add(key, value, stream_name="s")
    torch.testing.assert_close(direct.values(key), resumed.values(key), rtol=0, atol=0)
    torch.testing.assert_close(direct.channel_absmax(key), resumed.channel_absmax(key), rtol=0, atol=0)
    assert direct.seen_rows(key) == resumed.seen_rows(key)
    assert torch.equal(direct._generator.get_state(), resumed._generator.get_state())
