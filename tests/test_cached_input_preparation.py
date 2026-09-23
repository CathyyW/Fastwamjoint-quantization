from types import SimpleNamespace

import pytest
import torch

from fastwam_steerquant.adapters.fastwam import prepare_from_infer_kwargs


def fixture():
    @torch.inference_mode()
    def encode_image(**kwargs):
        return torch.ones(1, 2, 1, 2, 2)

    @torch.inference_mode()
    def append(**kwargs):
        context, mask = kwargs["context"], kwargs["context_mask"]
        return (torch.cat([context, torch.ones(1, 1, 4)], dim=1),
                torch.cat([mask, torch.ones(1, 1, dtype=torch.bool)], dim=1))

    def forbid_prompt(*args):
        raise AssertionError("Cached inputs must not access a text encoder")

    model = SimpleNamespace(device="cpu", torch_dtype=torch.float32,
        vae=SimpleNamespace(temporal_downsample_factor=4, upsampling_factor=2, model=SimpleNamespace(z_dim=2)),
        action_expert=SimpleNamespace(action_dim=3),
        _encode_input_image_latents_tensor=encode_image, encode_prompt=forbid_prompt,
        _append_proprio_to_context=append)
    kwargs = dict(input_image=torch.zeros(1, 3, 4, 4), proprio=torch.zeros(1, 3),
        prompt=None, context=torch.arange(8, dtype=torch.float32).reshape(2, 4),
        context_mask=torch.tensor([True, False]), num_video_frames=5, action_horizon=4)
    return model, kwargs


@pytest.mark.parametrize("batched", [True, False])
def test_cached_context_without_encoder_and_normal_autograd_tensors(batched):
    model, kwargs = fixture()
    if batched:
        kwargs["context"] = kwargs["context"].unsqueeze(0)
        kwargs["context_mask"] = kwargs["context_mask"].unsqueeze(0)
    source = kwargs["context"].clone()
    with torch.inference_mode():
        prepared = prepare_from_infer_kwargs(model, kwargs, seed=42)
    assert all(not torch.is_inference(value) for value in prepared.values())
    torch.testing.assert_close(kwargs["context"], source)
    assert prepared["context"].shape == (1, 3, 4)
    assert prepared["context_mask"].tolist() == [[True, False, True]]
    assert torch.equal(prepared["latents_video"][:, :, :1], prepared["first_frame_latents"])
    for name in ("first_frame_latents", "context"):
        x = torch.ones_like(prepared[name], requires_grad=True)
        (x * prepared[name]).sum().backward()
        assert torch.isfinite(x.grad).all()


@pytest.mark.parametrize("damage", ["both", "missing_mask", "missing_context", "batch", "length", "nan", "expert_cache"])
def test_invalid_cached_inputs_rejected(damage):
    model, kwargs = fixture()
    if damage == "both":
        kwargs["prompt"] = "task"
    elif damage == "missing_mask":
        del kwargs["context_mask"]
    elif damage == "missing_context":
        del kwargs["context"]
    elif damage == "batch":
        kwargs["context"] = kwargs["context"].unsqueeze(0).expand(2, -1, -1)
    elif damage == "length":
        kwargs["context_mask"] = torch.ones(3, dtype=torch.bool)
    elif damage == "nan":
        kwargs["context"][0, 0] = float("nan")
    else:
        kwargs["expert_cache"] = True
    with pytest.raises(ValueError):
        prepare_from_infer_kwargs(model, kwargs, seed=42)


def test_noise_remains_cpu_float32_independent_of_default_dtype():
    model, kwargs = fixture()
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        prepared = prepare_from_infer_kwargs(model, kwargs, seed=19)
    finally:
        torch.set_default_dtype(previous)
    expected = torch.randn(1, 4, 3, generator=torch.Generator(device="cpu").manual_seed(19), dtype=torch.float32)
    torch.testing.assert_close(prepared["latents_action"], expected, rtol=0, atol=0)
