"""Reusable input preparation for adapters using the existing FastWAM API."""
import torch


@torch.inference_mode(False)
@torch.no_grad()
def prepare_from_infer_kwargs(model, kwargs, *, seed):
    """kwargs must match production infer_action's ALREADY normalized inputs.

    input_image: batched CHW float in [-1,1], including correct camera packing.
    proprio: batched, normalized exactly as at training/deployment.
    Provide either prompt or a resolved context/context_mask pair. Cached padding
    semantics belong to the production adapter; this helper preserves the mask.
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
    if kwargs.get("expert_cache", False):
        raise ValueError("Calibration requires dense denoising with expert_cache=False.")
    prompt = kwargs.get("prompt")
    context, mask = kwargs.get("context"), kwargs.get("context_mask")
    use_context = context is not None or mask is not None
    if prompt is not None and use_context:
        raise ValueError("prompt and context/context_mask are mutually exclusive.")
    if prompt is None and (context is None or mask is None):
        raise ValueError("Provide prompt or both context/context_mask.")
    if prompt is not None:
        context, mask = model.encode_prompt(prompt)
    if context.ndim == 2:
        context = context.unsqueeze(0)
    if mask.ndim == 1:
        mask = mask.unsqueeze(0)
    if (context.ndim != 3 or mask.ndim != 2 or context.shape[0] != 1
            or tuple(mask.shape) != tuple(context.shape[:2])):
        raise ValueError("context/context_mask must be matching [1,L,D]/[1,L] tensors.")
    if not context.is_floating_point() or not torch.isfinite(context).all():
        raise ValueError("Context must contain finite floating-point embeddings.")
    context = context.to(device=model.device, dtype=model.torch_dtype).clone()
    mask = mask.to(device=model.device, dtype=torch.bool).clone()
    # VAE methods may use inference_mode. Clone outside that context before VJP.
    first = model._encode_input_image_latents_tensor(input_image=image, tiled=False).clone()
    context, mask = model._append_proprio_to_context(context=context, context_mask=mask, proprio=proprio)
    temporal_factor = int(model.vae.temporal_downsample_factor)
    frames = int(kwargs["num_video_frames"])
    if frames < 1 or (frames - 1) % temporal_factor or int(kwargs["action_horizon"]) < 1:
        raise ValueError("Invalid video frame count or action horizon.")
    latent_t = (frames - 1) // temporal_factor + 1
    h, w = image.shape[-2:]
    factor = int(model.vae.upsampling_factor)
    if h % factor or w % factor:
        raise ValueError("Input image dimensions must align to the VAE spatial factor.")
    if tuple(first.shape) != (1, model.vae.model.z_dim, 1, h // factor, w // factor):
        raise ValueError("Unexpected first-frame VAE latent shape.")
    # Production infer_action initializes two generators with the SAME seed.
    # Sharing one advancing generator would change the action-noise sample.
    generator = torch.Generator(device="cpu").manual_seed(seed)
    video = torch.randn((1, model.vae.model.z_dim, latent_t, h // factor, w // factor),
                        generator=generator, device="cpu", dtype=torch.float32).to(
                            device=image.device, dtype=model.torch_dtype)
    action = torch.randn((1, int(kwargs["action_horizon"]), model.action_expert.action_dim),
                         generator=torch.Generator(device="cpu").manual_seed(seed),
                         device="cpu", dtype=torch.float32).to(
                             device=image.device, dtype=model.torch_dtype)
    video[:, :, :1] = first
    return dict(latents_video=video, latents_action=action, first_frame_latents=first,
                context=context.clone(), context_mask=mask.clone())
