import copy

import pytest
import torch

from fastwam_steerquant.deployment import export_deployment, load_deployment, validate_payload
from fastwam_steerquant.runtime import WAMQuantLinear
from fastwam_steerquant.topology import resolve_site_module
from test_deployment import fixture_checkpoint, fixture_model


def aliased_model(device="cpu"):
    model = fixture_model(hidden=256, device=device)
    model.mot = torch.nn.Module()
    model.mot.mixtures = torch.nn.ModuleDict({"video": model.video_expert, "action": model.action_expert})
    model.dit = model.mot
    # Also exercise a directly registered Linear alias, not just shared parents.
    model.shortcut = model.video_expert.blocks[0].self_attn.q
    return model


@pytest.mark.parametrize("bits,rotation", [(8, "none"), (4, "none"), (4, "rht")])
@pytest.mark.parametrize("construct_device", ["cpu", "meta"])
def test_shared_experts_export_only_packed_and_strict_load(tmp_path, bits, rotation, construct_device):
    model = aliased_model()
    original = model.shortcut.weight.clone()
    cp = fixture_checkpoint(model, bits, rotation)
    path = export_deployment(model, cp, tmp_path / "deploy.pt", model_config={})
    payload = torch.load(path, weights_only=True)
    for spec in payload["sites"]:
        assert len(spec["aliases"]) >= 3
        canonical = payload["state_dict"][spec["module_name"] + ".weight"]
        assert canonical.dtype == torch.uint8
        for alias in spec["aliases"]:
            assert payload["state_dict"][alias + ".weight"] is canonical
    torch.testing.assert_close(model.shortcut.weight, original, rtol=0, atol=0)
    loaded, _, _ = load_deployment(path, lambda cfg, device: aliased_model(device),
                                   device="cpu", construct_device=construct_device)
    assert loaded.dit is loaded.mot
    assert loaded.mot.mixtures.video is loaded.video_expert
    assert loaded.shortcut is loaded.video_expert.blocks[0].self_attn.q
    for spec in payload["sites"]:
        module = resolve_site_module(loaded, spec["module_name"])
        assert isinstance(module, WAMQuantLinear)
        assert not list(module.parameters())
        assert module.weight.dtype == torch.uint8
        assert module.weight_scales.dtype == torch.float32
        for alias in spec["aliases"]:
            assert resolve_site_module(loaded, alias) is module
        for name, value in module.state_dict().items():
            torch.testing.assert_close(value, payload["state_dict"][spec["module_name"] + "." + name],
                                       rtol=0, atol=0)


@pytest.mark.parametrize("damage", ["missing", "bf16", "different_value", "duplicate", "overlap"])
def test_reject_alias_conflicts(tmp_path, damage):
    model = aliased_model()
    path = export_deployment(model, fixture_checkpoint(model), tmp_path / "deploy.pt", model_config={})
    payload = torch.load(path, weights_only=True)
    spec = payload["sites"][0]
    alias = next(a for a in spec["aliases"] if a != spec["module_name"])
    key = alias + ".weight"
    if damage == "missing":
        del payload["state_dict"][key]
    elif damage == "bf16":
        payload["state_dict"][key] = torch.ones(spec["out_features"], spec["in_features"], dtype=torch.bfloat16)
    elif damage == "different_value":
        payload["state_dict"][key] = payload["state_dict"][key].clone()
        payload["state_dict"][key][0, 0] ^= 1
    elif damage == "duplicate":
        spec["aliases"].append(alias)
    else:
        payload["sites"][1]["aliases"].append(alias)
    with pytest.raises(ValueError, match="alias"):
        validate_payload(payload)


def test_reject_builder_with_changed_alias_graph(tmp_path):
    model = aliased_model()
    path = export_deployment(model, fixture_checkpoint(model), tmp_path / "deploy.pt", model_config={})
    def builder(cfg, *, device):
        result = aliased_model(device)
        result.shortcut = copy.deepcopy(result.shortcut)
        return result
    with pytest.raises(ValueError, match="alias topology"):
        load_deployment(path, builder, device="cpu")


def test_nonaliased_legacy_v1_still_loads(tmp_path):
    model = fixture_model(hidden=32)
    path = export_deployment(model, fixture_checkpoint(model), tmp_path / "deploy.pt", model_config={})
    payload = torch.load(path, weights_only=True)
    for spec in payload["sites"]:
        del spec["aliases"]
    legacy = tmp_path / "legacy.pt"
    torch.save(payload, legacy)
    load_deployment(legacy, lambda cfg, device: fixture_model(32, device), device="cpu")
