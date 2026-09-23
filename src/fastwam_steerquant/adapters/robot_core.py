"""Dense real-robot FastWAMJoint core shared by VJP and deployed inference.

The robot source snapshot remains unchanged. This mixin only factors its dense
pre_dit -> MoT -> post_dit path out of the inference-only wrapper.
"""
import torch


class DenseJointCoreMixin:
    def _require_dense(self):
        for name in ("version1_sparse_controller", "action_future_attention_observer"):
            if getattr(self.mot, name, None) is not None:
                raise ValueError(f"SteerQuant dense adapter does not support {name}.")

    def _joint_denoise_core(self, *, latents_video, latents_action,
                           timestep_video, timestep_action, context, context_mask,
                           attention_mask, fuse_vae_embedding_in_latents,
                           action_condition=None):
        self._require_dense()
        if action_condition is not None:
            raise ValueError("FastWAMJoint calibration requires action_condition=None.")
        video = self.video_expert.pre_dit(
            x=latents_video, timestep=timestep_video, context=context,
            context_mask=context_mask, action=None,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents)
        action = self.action_expert.pre_dit(
            action_tokens=latents_action, timestep=timestep_action,
            context=context, context_mask=context_mask)
        pre = {"video": video, "action": action}
        out = self.mot(
            embeds_all={k: v["tokens"] for k, v in pre.items()},
            attention_mask=attention_mask,
            freqs_all={k: v["freqs"] for k, v in pre.items()},
            context_all={k: {"context": v["context"], "mask": v["context_mask"]}
                         for k, v in pre.items()},
            t_mod_all={k: v["t_mod"] for k, v in pre.items()})
        return (self.video_expert.post_dit(out["video"], video),
                self.action_expert.post_dit(out["action"], action))

    @torch.inference_mode()
    def _predict_joint_noise(self, latents_video, latents_action, timestep_video,
                             timestep_action, context, context_mask,
                             fuse_vae_embedding_in_latents, gt_action=None,
                             expert_cache_state=None, expert_cache_step=None):
        if expert_cache_state is not None or gt_action is not None:
            raise ValueError("Use dense joint inference without expert cache or GT action.")
        t, h, w = latents_video.shape[-3:]
        pt, ph, pw = self.video_expert.patch_size
        if t % pt or h % ph or w % pw:
            raise ValueError("Video latent shape does not align to patch size.")
        per_frame = (h // ph) * (w // pw)
        mask = self._build_mot_attention_mask(
            video_seq_len=(t // pt) * per_frame,
            action_seq_len=latents_action.shape[1],
            video_tokens_per_frame=per_frame, device=latents_video.device)
        return self._joint_denoise_core(
            latents_video=latents_video, latents_action=latents_action,
            timestep_video=timestep_video, timestep_action=timestep_action,
            context=context, context_mask=context_mask, attention_mask=mask,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
            action_condition=None)
