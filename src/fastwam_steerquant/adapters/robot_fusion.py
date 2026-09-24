"""Opt-in dense MoT post-block dispatch matching the simulation fusion path.

No edits to the hash-pinned robot source. Install on one model instance only,
before CUDA Graph capture. Fusion changes rounding and requires action checks.
"""
from types import MethodType


def _post_block(self, block, residual_x, mixed_attn_out, gate_msa, shift_mlp,
                scale_mlp, gate_mlp, context_payload, version1_layer_index=None,
                version1_expert_name=None):
    if self.training or block.training:
        raise RuntimeError("Robot block fusion is inference-only.")
    if any(getattr(self, key, None) is not None for key in
           ("version1_sparse_controller", "action_future_attention_observer")):
        raise ValueError("Robot block fusion requires the audited dense MoT route.")
    if getattr(block.self_attn.o, "supports_fused_gate_residual", False):
        x = block.self_attn.o.forward_gate_residual(mixed_attn_out, residual_x, gate_msa)
    else:
        x = block.gate(residual_x, gate_msa, block.self_attn.o(mixed_attn_out))
    if context_payload is not None and context_payload.get("context") is not None:
        mask = context_payload.get("mask")
        if mask is not None and mask.dim() == 3:
            mask = mask.unsqueeze(1)
        x = x + block.cross_attn(block.norm3(x), context_payload["context"], ctx_mask=mask)
    if getattr(block.ffn[0], "supports_fused_adaln", False):
        hidden = block.ffn[0].forward_adaln(x, scale_mlp, shift_mlp, block.norm2.eps)
    else:
        # Same operation order as qi.models.wan22.wan_video_dit.modulate.
        hidden = block.ffn[0](block.norm2(x) * (1 + scale_mlp) + shift_mlp)
    projected = block.ffn[1](hidden)
    if getattr(block.ffn[2], "supports_fused_gate_residual", False):
        return block.ffn[2].forward_gate_residual(projected, x, gate_mlp)
    return block.gate(x, gate_mlp, block.ffn[2](projected))


def install_robot_block_fusion_dispatch(model):
    if getattr(model, "_rollout_graph", None) is not None:
        raise RuntimeError("Install block fusion before CUDA Graph installation.")
    mot = model.mot
    if model.training or mot.training:
        raise RuntimeError("Robot block fusion is inference-only.")
    if any(getattr(mot, key, None) is not None for key in
           ("version1_sparse_controller", "action_future_attention_observer")):
        raise ValueError("Robot block fusion requires the audited dense MoT route.")
    if not callable(getattr(mot, "_apply_expert_post_block", None)):
        raise TypeError("Model does not expose the audited MoT post-block interface.")
    if getattr(mot, "_steerquant_block_fusion_dispatch", False):
        return
    mot._apply_expert_post_block = MethodType(_post_block, mot)
    mot._steerquant_block_fusion_dispatch = True
