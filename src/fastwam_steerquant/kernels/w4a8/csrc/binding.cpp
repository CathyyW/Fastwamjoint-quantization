#include <torch/extension.h>

#include <optional>

torch::Tensor w4a8_symmetric_dynamic_cuda(
    torch::Tensor x,
    torch::Tensor packed_weight,
    torch::Tensor weight_scales,
    std::optional<torch::Tensor> bias,
    std::optional<torch::Tensor> input_scale,
    bool input_scale_is_inverse);

torch::Tensor w4a8_symmetric_dynamic_adaln_cuda(
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

torch::Tensor w4a8_symmetric_dynamic_gate_residual_cuda(
    torch::Tensor x,
    torch::Tensor packed_weight,
    torch::Tensor weight_scales,
    std::optional<torch::Tensor> bias,
    std::optional<torch::Tensor> input_scale,
    bool input_scale_is_inverse,
    torch::Tensor residual,
    torch::Tensor gate,
    int64_t gate_span);

torch::Tensor w4a8_symmetric_static_cuda(
    torch::Tensor x,
    torch::Tensor packed_weight,
    torch::Tensor weight_scales,
    std::optional<torch::Tensor> bias,
    std::optional<torch::Tensor> input_scale,
    torch::Tensor activation_scale,
    torch::Tensor row_gains,
    bool input_scale_is_inverse);

torch::Tensor w4a8_symmetric_static_stream_cuda(
    torch::Tensor x,
    torch::Tensor packed_weight,
    torch::Tensor weight_scales,
    std::optional<torch::Tensor> bias,
    std::optional<torch::Tensor> input_scale,
    torch::Tensor activation_scale,
    torch::Tensor stream_gains,
    int64_t stream_span,
    bool input_scale_is_inverse);

torch::Tensor w4a8_symmetric_static_stream_scheduled_cuda(
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

torch::Tensor w4a8_symmetric_static_stream_scheduled_adaln_cuda(
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

torch::Tensor w4a8_symmetric_static_stream_scheduled_gate_residual_cuda(
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("symmetric_dynamic", &w4a8_symmetric_dynamic_cuda, "Packed W4A8 dynamic Linear (CUDA)");
  module.def(
      "symmetric_dynamic_adaln",
      &w4a8_symmetric_dynamic_adaln_cuda,
      "Packed W4A8 Linear with fused LayerNorm, AdaLN and dynamic A8 (CUDA)");
  module.def(
      "symmetric_dynamic_gate_residual",
      &w4a8_symmetric_dynamic_gate_residual_cuda,
      "Packed W4A8 Linear with fused gate and residual epilogue (CUDA)");
  module.def("symmetric_static", &w4a8_symmetric_static_cuda, "Packed W4A8 static Linear (CUDA)");
  module.def(
      "symmetric_static_stream",
      &w4a8_symmetric_static_stream_cuda,
      "Packed W4A8 static Linear with fused WAM stream mapping (CUDA)");
  module.def(
      "symmetric_static_stream_scheduled",
      &w4a8_symmetric_static_stream_scheduled_cuda,
      "Packed W4A8 static Linear with fused WAM stream and timestep mapping (CUDA)");
  module.def(
      "symmetric_static_stream_scheduled_adaln",
      &w4a8_symmetric_static_stream_scheduled_adaln_cuda,
      "Packed W4A8 WAM Linear with fused LayerNorm, AdaLN and static A8 (CUDA)");
  module.def(
      "symmetric_static_stream_scheduled_gate_residual",
      &w4a8_symmetric_static_stream_scheduled_gate_residual_cuda,
      "Packed W4A8 WAM Linear with fused gate and residual epilogue (CUDA)");
}
