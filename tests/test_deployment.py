from dataclasses import asdict, replace
import copy
import torch
import pytest
from fastwam_steerquant import QuantizationCheckpoint, QuantizedSite, FastWAMStreamConfig
from fastwam_steerquant.topology import enumerate_fastwamjoint_linears, resolve_site_module
from fastwam_steerquant.deployment import export_deployment, load_deployment, validate_payload
from fastwam_steerquant.runtime import apply_checkpoint, WAMQuantLinear
from fastwam_steerquant.rht import rht_signs
from test_pipeline import TinyFastWAM


def fixture_model(hidden=256, device="cpu"):
    with torch.device(device):
        model = TinyFastWAM(hidden=hidden).bfloat16()
        model.register_buffer("aux", torch.ones(3), persistent=False)
        model.extra = torch.nn.Linear(8, 8, dtype=torch.bfloat16)
    return model


def fixture_checkpoint(model, bits=8, rotation="none"):
    entries = []
    for site in enumerate_fastwamjoint_linears(model):
        n, k = site.module.weight.shape
        name = "context" if site.stream_kind == "context" else (
            "latent_frame_0" if site.expert == "video" else "action_chunk_0")
        entries.append(QuantizedSite(site.index, site.module_name, site.expert, site.operation, (name,),
            torch.randint(-7, 8, (n, k), dtype=torch.int8), torch.full((n, 1), .002),
            torch.linspace(.7, 1.4, k), torch.ones(2, 1), torch.tensor([2., 3.]), torch.ones(2, 1),
            0., 0., (0., 0.), (0., 0.),
            rht_signs(k, seed=42, module_name=site.module_name) if rotation == "rht" else None, rotation))
    return QuantizationCheckpoint(4, bits, 2, asdict(FastWAMStreamConfig(
        video_latent_frames=1, action_chunks=1, split_context=False)), {"epochs": 1}, {"epochs": 1}, tuple(entries))


@pytest.mark.parametrize("bits,rotation", [(8, "none"), (4, "none"), (4, "rht")])
@pytest.mark.parametrize("construct_device", ["meta", "cpu"])
def test_complete_export_direct_load_matches_existing_native_replacement(tmp_path, bits, rotation, construct_device):
    torch.manual_seed(5)
    original = fixture_model()
    cp = fixture_checkpoint(original, bits, rotation)
    path = export_deployment(original, cp, tmp_path / "deploy.pt", model_config={"hidden": 256})
    calls = []
    def builder(config, *, device):
        calls.append(device)
        return fixture_model(config["hidden"], device)
    loaded, state, metadata = load_deployment(path, builder, device="cpu", construct_device=construct_device)
    assert calls == [construct_device]
    assert state.num_calls == 2 and metadata["activation_bits"] == bits
    torch.testing.assert_close(loaded.aux, original.aux)
    torch.testing.assert_close(loaded.extra.weight, original.extra.weight)
    reference = copy.deepcopy(original)
    apply_checkpoint(reference, cp, backend=f"cutlass_w4a{bits}")
    for entry in cp.sites:
        a, b = resolve_site_module(loaded, entry.module_name), resolve_site_module(reference, entry.module_name)
        assert isinstance(a, WAMQuantLinear) and a.weight.dtype == torch.uint8
        assert a.weight.numel() * 2 == entry.qweight.numel()
        assert not list(a.parameters())  # No retained BF16 source Linear parameter.
        for name, value in b.state_dict().items():
            torch.testing.assert_close(a.state_dict()[name], value, rtol=0, atol=0)
    assert all(t.device.type == "cpu" for t in (*loaded.parameters(), *loaded.buffers()))


def test_export_rejects_partial_and_overwrite(tmp_path):
    model = fixture_model(hidden=32)
    cp = fixture_checkpoint(model)
    with pytest.raises(ValueError, match="every target"):
        export_deployment(model, replace(cp, sites=cp.sites[:1]), tmp_path / "x.pt", model_config={})
    path = export_deployment(model, cp, tmp_path / "x.pt", model_config={})
    with pytest.raises(FileExistsError):
        export_deployment(model, cp, path, model_config={})


def test_payload_rejects_unpacked_and_wrong_builder(tmp_path):
    model = fixture_model(hidden=32)
    path = export_deployment(model, fixture_checkpoint(model), tmp_path / "x.pt", model_config={})
    payload = torch.load(path, weights_only=True)
    key = payload["sites"][0]["module_name"] + ".weight"
    payload["state_dict"][key] = payload["state_dict"][key].to(torch.int8)
    with pytest.raises(ValueError, match="packed uint8"):
        validate_payload(payload)
    with pytest.raises(ValueError, match="shape/bias"):
        load_deployment(path, lambda cfg, device: fixture_model(64, device), device="cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("bits,rotation", [(8, "none"), (4, "rht")])
def test_native_direct_load_outputs_match_replacement(tmp_path, bits, rotation):
    original = fixture_model()
    cp = fixture_checkpoint(original, bits, rotation)
    path = export_deployment(original, cp, tmp_path / "gpu.pt", model_config={})
    direct, direct_state, _ = load_deployment(path, lambda cfg, device: fixture_model(device=device))
    reference = original.cuda()
    reference_state, _ = apply_checkpoint(reference, cp, backend=f"cutlass_w4a{bits}")
    for call in (0, 1):
        direct_state.set(call); reference_state.set(call)
        for entry in cp.sites:
            x = torch.randn(1, 6, entry.qweight.shape[1], device="cuda", dtype=torch.bfloat16)
            a = resolve_site_module(direct, entry.module_name)(x)
            b = resolve_site_module(reference, entry.module_name)(x)
            torch.testing.assert_close(a, b, atol=0, rtol=0)
