"""Reusable input preparation for adapters using the existing FastWAM API."""
import torch


@torch.no_grad()
def prepare_from_infer_kwargs(model, kwargs, *, seed):
    """kwargs must match production infer_action's ALREADY normalized inputs.

    input_image: batched CHW float in [-1,1], including correct camera packing.
    proprio: batched, normalized exactly as at training/deployment.
    This function does not guess camera order, resize or normalize raw robot data.
    """
    image = kwargs["input_image"].to(device=model.device, dtype=model.torch_dtype)
    proprio = kwargs["proprio"].to(device=model.device, dtype=model.torch_dtype)
    if image.ndim != 4 or image.shape[0] != 1 or proprio.ndim != 2 or proprio.shape[0] != 1:
        raise ValueError("Calibration currently expects batch size one, BCHW images and batched proprio.")
    if not image.is_floating_point() or not torch.isfinite(image).all() or image.abs().max() > 1.001:
        raise ValueError("Provide production-normalized image values in [-1,1].")
    if float(kwargs.get("text_cfg_scale", 1.0)) != 1:
        raise ValueError("This calibration preparation supports text_cfg_scale=1.")
    if kwargs.get("rand_device", "cpu") != "cpu" or kwargs.get("tiled", False):
        raise ValueError("This helper expects CPU noise generation and tiled=False.")
    first = model._encode_input_image_latents_tensor(input_image=image, tiled=False)
    context, mask = model.encode_prompt(kwargs["prompt"])
    context, mask = model._append_proprio_to_context(context=context, context_mask=mask, proprio=proprio)
    latent_t = (int(kwargs["num_video_frames"]) - 1) // int(model.vae.temporal_downsample_factor) + 1
    h, w = image.shape[-2:]
    factor = int(model.vae.upsampling_factor)
    # Production infer_action initializes two generators with the SAME seed.
    # Sharing one advancing generator would change the action-noise sample.
    generator = torch.Generator(device="cpu").manual_seed(seed)
    video = torch.randn((1, model.vae.model.z_dim, latent_t, h // factor, w // factor),
                        generator=generator).to(device=image.device, dtype=model.torch_dtype)
    action = torch.randn((1, int(kwargs["action_horizon"]), model.action_expert.action_dim),
                         generator=torch.Generator(device="cpu").manual_seed(seed)).to(
                             device=image.device, dtype=model.torch_dtype)
    video[:, :, :1] = first
    return dict(latents_video=video, latents_action=action, first_frame_latents=first,
                context=context, context_mask=mask)
