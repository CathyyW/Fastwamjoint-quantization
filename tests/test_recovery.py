from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from fastwam_steerquant import (
    ActivationCache, SensitivityAccumulator, SensitivityField,
    DCalibrationConfig, GammaCalibrationConfig, FastWAMStreamConfig,
    calibrate_model, enumerate_fastwamjoint_linears,
)
from fastwam_steerquant.cache import CacheKey
from fastwam_steerquant.sensitivity import SiteSensitivity
from fastwam_steerquant.recovery import atomic_snapshot
from test_pipeline import TinyFastWAM

ROOT = Path(__file__).resolve().parents[1]


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_source_dtype_reservoir_and_rng_resume_are_exact(tmp_path, dtype):
    torch.manual_seed(17)
    reference = ActivationCache(num_calls=1, max_rows_per_cell=7, seed=19)
    compact = ActivationCache(num_calls=1, max_rows_per_cell=7, seed=19, preserve_source_dtype=True)
    key = CacheKey(0, 0, 0)
    for i in range(5):
        value = torch.randn(11, 16).to(dtype)
        reference.add(key, value, stream_name="stream")
        compact.add(key, value, stream_name="stream")
        if i == 1:
            atomic_snapshot({"cache": compact.state_dict(), "completed": 2}, tmp_path / "progress.pt")
            saved = torch.load(tmp_path / "progress.pt", weights_only=True)
            compact = ActivationCache.from_state_dict(saved["cache"])
        torch.testing.assert_close(compact.values(key), reference.values(key), rtol=0, atol=0)
        torch.testing.assert_close(compact.channel_absmax(key), reference.channel_absmax(key), rtol=0, atol=0)
        assert compact.seen_rows(key) == reference.seen_rows(key)
    assert compact.state_dict()["cells"][0]["values"].dtype == dtype
    assert not (tmp_path / "progress.pt.pending").exists()


def test_accumulator_resume_preserves_exact_counts_and_squares():
    site = enumerate_fastwamjoint_linears(TinyFastWAM())[0]
    a = SensitivityAccumulator(num_calls=2)
    b = SensitivityAccumulator(num_calls=2)
    for index in range(5):
        value = torch.full((2, 1), 0.123 * index)
        a.add(site, ("stream",), value)
        b.add(site, ("stream",), value)
        if index == 2:
            b = SensitivityAccumulator.from_state_dict(b.state_dict())
    torch.testing.assert_close(a.finalize().sites[0].values, b.finalize().sites[0].values, rtol=0, atol=0)
    assert (b.finalize().sites[0].observation_counts == 5).all()


def test_calibration_resumes_committed_sites_without_reoptimizing():
    torch.manual_seed(2)
    sites = enumerate_fastwamjoint_linears(TinyFastWAM())[:2]
    cache = ActivationCache(num_calls=1, max_rows_per_cell=8, preserve_source_dtype=True)
    fields = []
    for site in sites:
        cache.add(CacheKey(site.index, 0, 0), torch.randn(8, 4).bfloat16(), stream_name="latent_frame_0")
        fields.append(SiteSensitivity(site.index, site.module_name, ("latent_frame_0",),
                                      torch.ones(1, 1), torch.ones(1, 1, dtype=torch.long)))
    kwargs = dict(cache=cache, sensitivity_field=SensitivityField(1, tuple(fields)),
                  stream_config=FastWAMStreamConfig(video_latent_frames=1),
                  d_config=DCalibrationConfig(epochs=0), gamma_config=GammaCalibrationConfig(epochs=0))
    committed = {}
    def interrupted(entry):
        committed[entry.site_index] = entry
        raise RuntimeError("simulated interruption after commit")
    with pytest.raises(RuntimeError, match="simulated"):
        calibrate_model(sites, on_site_complete=interrupted, **kwargs)
    callbacks = []
    recovered = calibrate_model(sites, resume_sites=committed, on_site_complete=lambda e: callbacks.append(e.site_index), **kwargs)
    direct = calibrate_model(sites, **kwargs)
    assert callbacks == [sites[1].index]
    for a, b in zip(recovered.sites, direct.sites):
        for field in ("qweight", "weight_scales", "input_scale", "gamma_gains", "clipping_ranges"):
            torch.testing.assert_close(getattr(a, field), getattr(b, field), rtol=0, atol=0)


def test_single_shard_merge_shares_storage(tmp_path, monkeypatch):
    cache = ActivationCache(num_calls=1, max_rows_per_cell=4)
    cache.add(CacheKey(0, 0, 0), torch.ones(2, 3), stream_name="stream")
    source = cache.save(tmp_path / "source.pt")
    target = tmp_path / "merged.pt"
    monkeypatch.setattr(sys, "argv", ["merge", "cache", "--inputs", str(source), "--output", str(target)])
    script("merge_artifacts").main()
    assert source.stat().st_ino == target.stat().st_ino
    assert ActivationCache.load(target).keys == cache.keys


def test_supervisor_recycles_and_stops_on_nonretryable_error(tmp_path):
    state = tmp_path / "counter"
    code = ("from pathlib import Path; import sys; p=Path(sys.argv[1]); "
            "n=int(p.read_text()) if p.exists() else 0; p.write_text(str(n+1)); "
            "sys.exit(75 if n < 2 else 0)")
    base = [sys.executable, str(ROOT / "scripts/resumable_worker.py"), "--max-recycles", "3", "--"]
    result = subprocess.run([*base, sys.executable, "-c", code, str(state)], capture_output=True)
    assert result.returncode == 0 and state.read_text() == "3"
    result = subprocess.run([*base, sys.executable, "-c", "raise SystemExit(2)"], capture_output=True)
    assert result.returncode == 2
