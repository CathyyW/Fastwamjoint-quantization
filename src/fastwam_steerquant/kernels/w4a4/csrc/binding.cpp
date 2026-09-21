// Migrated from Cosmos-Policy's Apache-2.0 W4A4 extension binding.
#include <torch/extension.h>

#include <optional>

torch::Tensor w4a4_dynamic_per_token_cuda(
    torch::Tensor x,
    torch::Tensor packed_weight,
    torch::Tensor weight_scales,
    std::optional<torch::Tensor> bias,
    double clip_ratio);

torch::Tensor w4a4_rht_dynamic_per_token_cuda(
    torch::Tensor x,
    torch::Tensor rotation_signs,
    torch::Tensor packed_weight,
    torch::Tensor weight_scales,
    std::optional<torch::Tensor> bias,
    double clip_ratio);

torch::Tensor w4a4_symmetric_dynamic_cuda(
    torch::Tensor x,
    torch::Tensor packed_weight,
    torch::Tensor weight_scales,
    std::optional<torch::Tensor> bias,
    std::optional<torch::Tensor> input_scale,
    bool input_scale_is_inverse);

torch::Tensor w4a4_symmetric_dynamic_adaln_cuda(
    torch::Tensor x,
    torch::Tensor adaln_scale,
    torch::Tensor adaln_shift,
    int64_t modulation_span,
    double epsilon,
    torch::Tensor packed_weight,
    torch::Tensor weight_scales,
    std::optional<torch::Tensor> bias,
    std::optional<torch::Tensor> input_scale,
    bool input_scale_is_inverse);

torch::Tensor w4a4_symmetric_dynamic_gate_residual_cuda(
    torch::Tensor x,
    torch::Tensor packed_weight,
    torch::Tensor weight_scales,
    std::optional<torch::Tensor> bias,
    std::optional<torch::Tensor> input_scale,
    bool input_scale_is_inverse,
    torch::Tensor residual,
    torch::Tensor gate,
    int64_t gate_span);

torch::Tensor w4a4_symmetric_static_cuda(
    torch::Tensor x,
    torch::Tensor packed_weight,
    torch::Tensor weight_scales,
    std::optional<torch::Tensor> bias,
    std::optional<torch::Tensor> input_scale,
    torch::Tensor activation_scale,
    torch::Tensor row_gains,
    bool input_scale_is_inverse);

torch::Tensor w4a4_symmetric_static_stream_scheduled_cuda(
    torch::Tensor x,
    torch::Tensor packed_weight,
    torch::Tensor weight_scales,
    std::optional<torch::Tensor> bias,
    std::optional<torch::Tensor> input_scale,
    torch::Tensor activation_scales,
    torch::Tensor stream_gains,
    int64_t stream_span,
    int64_t schedule_index,
    bool input_scale_is_inverse);

torch::Tensor w4a4_symmetric_static_stream_scheduled_rht_cuda(
    torch::Tensor x,
    torch::Tensor rotation_signs,
    torch::Tensor packed_weight,
    torch::Tensor weight_scales,
    std::optional<torch::Tensor> bias,
    std::optional<torch::Tensor> input_scale,
    torch::Tensor activation_scales,
    torch::Tensor stream_gains,
    int64_t stream_span,
    int64_t schedule_index,
    bool input_scale_is_inverse);

torch::Tensor w4a4_symmetric_static_stream_scheduled_adaln_cuda(
    torch::Tensor x,
    torch::Tensor adaln_scale,
    torch::Tensor adaln_shift,
    int64_t modulation_span,
    double epsilon,
    torch::Tensor packed_weight,
    torch::Tensor weight_scales,
    std::optional<torch::Tensor> bias,
    std::optional<torch::Tensor> input_scale,
    torch::Tensor activation_scales,
    torch::Tensor stream_gains,
    int64_t stream_span,
    int64_t schedule_index,
    bool input_scale_is_inverse);

torch::Tensor w4a4_symmetric_static_stream_scheduled_adaln_rht_cuda(
    torch::Tensor x,
    torch::Tensor adaln_scale,
    torch::Tensor adaln_shift,
    int64_t modulation_span,
    double epsilon,
    torch::Tensor rotation_signs,
    torch::Tensor packed_weight,
    torch::Tensor weight_scales,
    std::optional<torch::Tensor> bias,
    std::optional<torch::Tensor> input_scale,
    torch::Tensor activation_scales,
    torch::Tensor stream_gains,
    int64_t stream_span,
    int64_t schedule_index,
    bool input_scale_is_inverse);

torch::Tensor w4a4_symmetric_static_stream_scheduled_gate_residual_cuda(
    torch::Tensor x,
    torch::Tensor packed_weight,
    torch::Tensor weight_scales,
    std::optional<torch::Tensor> bias,
    std::optional<torch::Tensor> input_scale,
    torch::Tensor activation_scales,
    torch::Tensor stream_gains,
    int64_t stream_span,
    int64_t schedule_index,
    bool input_scale_is_inverse,
    torch::Tensor residual,
    torch::Tensor gate,
    int64_t gate_span);

torch::Tensor w4a4_symmetric_static_stream_scheduled_gate_residual_rht_cuda(
    torch::Tensor x,
    torch::Tensor rotation_signs,
    torch::Tensor packed_weight,
    torch::Tensor weight_scales,
    std::optional<torch::Tensor> bias,
    std::optional<torch::Tensor> input_scale,
    torch::Tensor activation_scales,
    torch::Tensor stream_gains,
    int64_t stream_span,
    int64_t schedule_index,
    bool input_scale_is_inverse,
    torch::Tensor residual,
    torch::Tensor gate,
    int64_t gate_span);

torch::Tensor w4a4_rht_official_dynamic_per_token_cuda(
    torch::Tensor x, torch::Tensor signs, torch::Tensor weight,
    torch::Tensor scales, std::optional<torch::Tensor> bias, double clip_ratio);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("rht_official_dynamic_per_token", &w4a4_rht_official_dynamic_per_token_cuda,
             "Official had12/had28 fused rotation, A4 packing and S4 MMA");
  module.def(
      "dynamic_per_token",
      &w4a4_dynamic_per_token_cuda,
      "Native S4xS4 dynamic per-token Linear (CUDA)");
  module.def(
      "rht_dynamic_per_token",
      &w4a4_rht_dynamic_per_token_cuda,
      "Fused randomized Hadamard, dynamic A4 quantization and S4xS4 Linear (CUDA)");
  module.def("symmetric_dynamic", &w4a4_symmetric_dynamic_cuda, "Packed W4A4 dynamic Linear (CUDA)");
  module.def(
      "symmetric_dynamic_adaln",
      &w4a4_symmetric_dynamic_adaln_cuda,
      "Packed W4A4 Linear with fused LayerNorm, AdaLN and dynamic A4 (CUDA)");
  module.def(
      "symmetric_dynamic_gate_residual",
      &w4a4_symmetric_dynamic_gate_residual_cuda,
      "Packed W4A4 Linear with fused gate and residual epilogue (CUDA)");
  module.def("symmetric_static", &w4a4_symmetric_static_cuda, "Packed W4A4 static Linear (CUDA)");
  module.def(
      "symmetric_static_stream_scheduled",
      &w4a4_symmetric_static_stream_scheduled_cuda,
      "Packed W4A4 static Linear with fused WAM stream and timestep mapping (CUDA)");
  module.def(
      "symmetric_static_stream_scheduled_rht",
      &w4a4_symmetric_static_stream_scheduled_rht_cuda,
      "Packed W4A4 WAM Linear with fused RHT and static A4 (CUDA)");
  module.def(
      "symmetric_static_stream_scheduled_adaln",
      &w4a4_symmetric_static_stream_scheduled_adaln_cuda,
      "Packed W4A4 WAM Linear with fused LayerNorm, AdaLN and static A4 (CUDA)");
  module.def(
      "symmetric_static_stream_scheduled_adaln_rht",
      &w4a4_symmetric_static_stream_scheduled_adaln_rht_cuda,
      "Packed W4A4 WAM Linear with fused LayerNorm, AdaLN, RHT and static A4 (CUDA)");
  module.def(
      "symmetric_static_stream_scheduled_gate_residual",
      &w4a4_symmetric_static_stream_scheduled_gate_residual_cuda,
      "Packed W4A4 WAM Linear with fused gate and residual epilogue (CUDA)");
  module.def(
      "symmetric_static_stream_scheduled_gate_residual_rht",
      &w4a4_symmetric_static_stream_scheduled_gate_residual_rht_cuda,
      "Packed W4A4 WAM RHT Linear with fused gate and residual epilogue (CUDA)");
}
