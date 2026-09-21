from __future__ import annotations

import copy
from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from fastwam_steerquant import (
    ActivationCache, ActivationCollector, DCalibrationConfig, DenoiseCallState,
    FastWAMStreamConfig, GammaCalibrationConfig, QuantizationCheckpoint,
    SensitivityField, calibrate_model, enumerate_fastwamjoint_linears,
)
from fastwam_steerquant.rht import rht_signs, rht_transform
from fastwam_steerquant.cache import CacheKey
from fastwam_steerquant.checkpoint import QuantizedSite
from fastwam_steerquant.runtime import WAMQuantLinear
from fastwam_steerquant.sensitivity import SiteSensitivity
from fastwam_steerquant.rht_install import install_rht_, rotation_identity
from test_pipeline import TinyFastWAM


def test_rotation_install_preserves_function_gradient_and_cache_order():
    torch.manual_seed(29)
    model = TinyFastWAM()
    original = copy.deepcopy(model)
    install_rht_(model, seed=42)
    sites = enumerate_fastwamjoint_linears(model)
    originals = enumerate_fastwamjoint_linears(original)
    state = DenoiseCallState(1)
    cache = ActivationCache(num_calls=1, max_rows_per_cell=128)
    config = FastWAMStreamConfig(video_latent_frames=1, action_chunks=1, split_context=False)
    with ActivationCollector(sites, state=state, stream_config=config, cache=cache):
        state.set(0)
        for site, reference in zip(sites, originals):
            x = torch.randn(1, 4, site.module.in_features, requires_grad=True)
            y = site.module(x)
            expected = reference.module(x)
            torch.testing.assert_close(y, expected, atol=2e-6, rtol=2e-5)
            a, = torch.autograd.grad(y.square().sum(), x, retain_graph=True)
            b, = torch.autograd.grad(expected.square().sum(), x)
            torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)
            rotated = rht_transform(x.detach(), site.module._wam_rotation_signs, "rht")
            torch.testing.assert_close(cache.values(CacheKey(site.index, 0, 0)), rotated.reshape(-1, x.shape[-1]))
    with pytest.raises(ValueError, match="already installed"):
        install_rht_(model)


def test_rotated_calibration_roundtrip_resume_and_cache_identity(tmp_path):
    torch.manual_seed(11)
    model = TinyFastWAM()
    install_rht_(model)
    site = enumerate_fastwamjoint_linears(model)[0]
    cache = ActivationCache(num_calls=1, max_rows_per_cell=8)
    cache.rotation_config = rotation_identity("rht", 42)
    cache.add(CacheKey(0, 0, 0), torch.randn(8, 4), stream_name="latent_frame_0")
    field = SensitivityField(1, (SiteSensitivity(0, site.module_name, ("latent_frame_0",),
                                               torch.ones(1, 1), torch.ones(1, 1, dtype=torch.long)),), cache.rotation_config)
    restored_field = SensitivityField.load(field.save(tmp_path / "sensitivity.pt"))
    assert SensitivityField.merge([field, restored_field]).rotation_config == cache.rotation_config
    with pytest.raises(ValueError, match="rotation domains"):
        SensitivityField.merge([field, replace(field, rotation_config={})])
    config = FastWAMStreamConfig(video_latent_frames=1)
    kwargs = dict(cache=cache, sensitivity_field=field, stream_config=config,
                  d_config=DCalibrationConfig(activation_bits=4, epochs=1),
                  gamma_config=GammaCalibrationConfig(activation_bits=4, epochs=1))
    cp = calibrate_model([site], **kwargs)
    loaded = QuantizationCheckpoint.load(cp.save(tmp_path / "rotated.pt"))
    assert loaded.state_dict()["format"] == "fastwam_steerquant_calibration_rht_v1"
    bad = loaded.state_dict(); bad["format"] = "fastwam_steerquant_calibration_v1"
    with pytest.raises(ValueError, match="format and rotation"):
        QuantizationCheckpoint.from_state_dict(bad)
    entry = loaded.sites[0]
    assert entry.rotation == "rht"
    torch.testing.assert_close(entry.input_rotation_signs, site.module._wam_rotation_signs)
    resumed = calibrate_model([site], resume_sites={0: entry}, **kwargs)
    torch.testing.assert_close(resumed.sites[0].qweight, entry.qweight)
    with pytest.raises(ValueError, match="rotation"):
        calibrate_model([site], resume_sites={0: replace(entry, rotation="none", input_rotation_signs=None)}, **kwargs)
    restored_cache = ActivationCache.load(cache.save(tmp_path / "cache.pt"))
    assert ActivationCache.merge([cache, restored_cache]).rotation_config == cache.rotation_config
    with pytest.raises(ValueError, match="configurations"):
        ActivationCache.merge([cache, ActivationCache(num_calls=1, max_rows_per_cell=8)])
    state = DenoiseCallState(1); state.set(0)
    module = WAMQuantLinear(site.module, entry, state=state, stream_config=config,
                            weight_bits=4, activation_bits=4)
    x = torch.randn(1, 4, 4)
    rotated = rht_transform(x, entry.input_rotation_signs, "rht")
    scale = entry.clipping_ranges[0] / 7
    qx = (rotated / entry.input_scale / scale).round().clamp(-7, 7) * scale
    expected = F.linear(qx, entry.qweight.float() * entry.weight_scales, site.module.bias)
    torch.testing.assert_close(module(x), expected)


CASES = [(32, 1024, 3072), (32, 3072, 1024), (32, 1024, 4096),
         (32, 4096, 1024), (129, 1024, 3072), (129, 3072, 3072),
         (294, 3072, 3072), (294, 3072, 14336), (294, 14336, 3072)]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("m,k,n", CASES)
def test_official_fused_wam_shapes_runtime_and_graph(dtype, m, k, n):
    from fastwam_steerquant.kernels.w4a4 import wam_ops, pack_signed_int4
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    counts = (128, 1) if m == 129 else (16, 16) if m == 32 else (98, 98, 98)
    names = ("text", "proprio") if m == 129 else tuple(
        f"action_chunk_{i}" if m == 32 else f"latent_frame_{i}" for i in range(len(counts)))
    source = torch.nn.Linear(k, n, device="cuda", dtype=dtype)
    entry = QuantizedSite(0, "linear", "video" if m == 294 else "action",
        "cross_attn.k" if m == 129 else "self_attn.q", names,
        torch.randint(-7, 8, (n, k), dtype=torch.int8), torch.full((n, 1), .002),
        torch.linspace(.5, 2, k), torch.ones(10, len(counts)),
        torch.linspace(2., 4., 10), torch.tensor(counts).float()[None].repeat(10, 1) / m,
        0., 0., (0.,)*10, (0.,)*10,
        rht_signs(k, seed=42, module_name="linear"), "rht")
    # Nontrivial gains obey the same weighted-log conservation as calibration.
    raw = torch.linspace(-.4, .4, len(counts))
    entry.gamma_gains[:] = (raw - (raw * entry.token_fractions[0]).sum()).exp()
    state = DenoiseCallState(10)
    module = WAMQuantLinear(source, entry, state=state, stream_config=FastWAMStreamConfig(),
                            weight_bits=4, activation_bits=4, backend="cutlass_w4a4")
    x = torch.randn(2 if m == 129 else 1, m, k, device="cuda", dtype=dtype)
    def oracle():
        z = rht_transform(x.float(), module.input_rotation_signs, "rht").to(dtype).float()
        gain = torch.repeat_interleave(module.gamma_gains[state.require()], torch.tensor(counts, device="cuda"))[None, :, None]
        scale = module.clipping_ranges[state.require()]
        qa = (z * module.input_scale * (gain / scale)).round().clamp(-7, 7)
        acc = F.linear(qa, entry.qweight.cuda().float())
        return (acc * (scale / gain) * module.weight_scales + source.bias.float()).to(dtype)
    with torch.inference_mode():
        for call in (0, 9):
            state.set(call)
            actual = module(x)
            expected = oracle()
            rel = (actual.float()-expected.float()).square().mean().sqrt()/expected.float().square().mean().sqrt()
            assert rel < .003, float(rel)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph): captured = module(x)
            graph.replay()
            torch.testing.assert_close(captured, actual, atol=0, rtol=0)
            x.mul_(.79)
            eager = module(x)
            graph.replay()
            torch.testing.assert_close(captured, eager, atol=0, rtol=0)
    with pytest.raises(ValueError, match="block fusion"):
        WAMQuantLinear(source, entry, state=state, stream_config=FastWAMStreamConfig(),
                       weight_bits=4, activation_bits=4, backend="cutlass_w4a4", fuse_block=True)
