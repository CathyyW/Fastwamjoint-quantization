// Migrated from Cosmos-Policy's Apache-2.0 W4A4 backend and modified for
// FastWAMJoint block-Hadamard widths. CUTLASS and Fast Hadamard Transform
// headers retain their upstream BSD-3-Clause licenses under third_party/.
#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <algorithm>
#include <optional>
#include <type_traits>
#include <cstdlib>
#include <cstring>

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/threadblock/default_epilogue_tensor_op.h"
#include "cutlass/epilogue/threadblock/epilogue_with_visitor_callbacks.h"
#include "cutlass/epilogue/threadblock/fusion/visitors.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/default_gemm_universal_with_visitor.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"
#include "fast_hadamard_transform_common.h"

namespace {

using ElementA = cutlass::int4b_t;
using ElementB = cutlass::int4b_t;
using ElementAccumulator = int32_t;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 256>;
using WarpShape = cutlass::gemm::GemmShape<64, 64, 256>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 64>;
using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

static int const kAlignmentA = 32;
static int const kAlignmentB = 32;
static int const kMainloopStages = 3;

// Load a compact [groups, N] AdaLN gate directly in the GEMM epilogue. Rows
// within one spatial/temporal group share a gate row, so expanding the gate to
// [M, N] would waste both memory bandwidth and a pointwise kernel launch.
template <class ThreadMap, class Element>
struct VisitorGroupedRowBroadcast {
  struct Arguments {
    Element const* ptr_gate = nullptr;
    Element null_default = Element(1);
    int64_t group_span = 1;
    int64_t gate_row_stride = 0;
  };
  using Params = Arguments;

  template <class ProblemShape>
  static constexpr Params to_underlying_arguments(ProblemShape const&, Arguments const& args, void*) {
    return args;
  }
  template <class ProblemShape>
  static size_t get_workspace_size(ProblemShape const&, Arguments const&) { return 0; }

  static const int Stages = ThreadMap::Stages;
  struct SharedStorage {};
  static int constexpr vec_bits = ThreadMap::kElementsPerAccess * cutlass::sizeof_bits<Element>::value;
  using VecType = cutlass::uint_bit_t<cute::min(128, vec_bits)>;
  static int constexpr VecLength = sizeof(VecType) / sizeof(Element);

  CUTLASS_HOST_DEVICE VisitorGroupedRowBroadcast() {}
  CUTLASS_HOST_DEVICE VisitorGroupedRowBroadcast(Params const& params, SharedStorage const&)
      : params_ptr(&params) {}
  Params const* params_ptr;

  template <class RTensor, class CTensor, class ProblemShape>
  struct Callbacks : cutlass::epilogue::threadblock::EmptyCallbacks {
    CUTLASS_DEVICE Callbacks(
        RTensor&& tC_rGate,
        CTensor&& tC_cGate,
        ProblemShape problem_shape,
        Params const* params_ptr)
        : tC_rGate(cute::forward<RTensor>(tC_rGate)),
          tC_cGate(cute::forward<CTensor>(tC_cGate)),
          problem_shape(problem_shape),
          params_ptr(params_ptr) {}

    RTensor tC_rGate;
    CTensor tC_cGate;
    ProblemShape problem_shape;
    Params const* params_ptr;

    CUTLASS_DEVICE void begin_step(int step_idx) {
      cute::clear(tC_rGate(cute::_, cute::_, cute::_, step_idx % Stages));
      auto coord_v = cute::filter(tC_cGate(cute::_, cute::_, cute::_, step_idx));
      auto dst_v = cute::filter(tC_rGate(cute::_, cute::_, cute::_, step_idx % Stages));
      if (params_ptr->ptr_gate == nullptr) {
        auto dst_elements = cute::recast<Element>(dst_v);
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < cute::size(dst_elements); ++i) dst_elements(i) = params_ptr->null_default;
        return;
      }
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < cute::size(dst_v); ++i) {
        auto coord = coord_v(i);
        int64_t row = cute::get<0>(coord);
        int64_t column = cute::get<1>(coord);
        bool guard = cute::elem_less(coord, problem_shape);
        Element const* source = params_ptr->ptr_gate +
            (row / params_ptr->group_span) * params_ptr->gate_row_stride + column;
        cutlass::arch::global_load<VecType, sizeof(VecType)>(dst_v(i), source, guard);
      }
    }

    template <class ElementAccumulator_, int FragmentSize>
    CUTLASS_DEVICE auto visit(
        int iter_idx,
        int,
        int,
        int frg_idx,
        cutlass::Array<ElementAccumulator_, FragmentSize> const&) {
      auto fragments = cute::recast<cutlass::Array<Element, FragmentSize>>(
          cute::coalesce(tC_rGate(cute::_, cute::_, cute::_, iter_idx % Stages)));
      return fragments(frg_idx);
    }
  };

  template <class ProblemShape>
  CUTLASS_DEVICE auto get_callbacks(
      cutlass::gemm::GemmCoord threadblock_tile_offset,
      int thread_idx,
      ProblemShape problem_shape) {
    auto mGate = cute::make_tensor(
        cute::make_gmem_ptr(params_ptr->ptr_gate),
        problem_shape,
        cute::make_stride(int64_t(cute::get<1>(problem_shape)), cute::_1{}, cute::_0{}));
    auto tC_gGate = cute::recast<VecType>(cute::group_modes<3, 6>(
        ThreadMap::partition(mGate, thread_idx, threadblock_tile_offset)));
    auto tC_rGate = cute::make_tensor<VecType>(cute::make_layout(cute::flatten(cute::make_shape(
        cute::take<0, 3>(tC_gGate.shape()), cute::Int<Stages>{}))));
    auto cGate = cute::make_identity_tensor(mGate.shape());
    auto tC_cGate = cute::outer_partition(
        cute::group_modes<3, 6>(ThreadMap::partition(cGate, thread_idx, threadblock_tile_offset)),
        cute::Shape<cute::Int<VecLength>>{},
        (cute::_0{}));
    return Callbacks<decltype(tC_rGate), decltype(tC_cGate), ProblemShape>(
        cute::move(tC_rGate), cute::move(tC_cGate), problem_shape, params_ptr);
  }
};

// CUTLASS's stock VisitorAuxLoad assumes a non-null pointer. This variant
// supplies zero for ordinary Linears and loads the residual for DiT output
// projections, allowing both paths to share one epilogue dataflow.
template <class ThreadMap, class Element, class StrideMNL>
struct VisitorOptionalAuxLoad {
  struct Arguments {
    Element* ptr_aux = nullptr;
    Element null_default = Element(0);
    StrideMNL dAux = {};
  };
  using Params = Arguments;

  template <class ProblemShape>
  static constexpr Params to_underlying_arguments(ProblemShape const&, Arguments const& args, void*) {
    return args;
  }
  template <class ProblemShape>
  static size_t get_workspace_size(ProblemShape const&, Arguments const&) { return 0; }

  static const int Stages = ThreadMap::Stages;
  struct SharedStorage {};
  static int constexpr vec_bits = ThreadMap::kElementsPerAccess * cutlass::sizeof_bits<Element>::value;
  using VecType = cutlass::uint_bit_t<cute::min(128, vec_bits)>;
  static int constexpr VecLength = sizeof(VecType) / sizeof(Element);

  CUTLASS_HOST_DEVICE VisitorOptionalAuxLoad() {}
  CUTLASS_HOST_DEVICE VisitorOptionalAuxLoad(Params const& params, SharedStorage const&)
      : params_ptr(&params) {}
  Params const* params_ptr;

  template <class GTensor, class RTensor, class CTensor, class ProblemShape>
  struct Callbacks : cutlass::epilogue::threadblock::EmptyCallbacks {
    CUTLASS_DEVICE Callbacks(
        GTensor&& tC_gAux,
        RTensor&& tC_rAux,
        CTensor&& tC_cAux,
        ProblemShape problem_shape,
        Params const* params_ptr)
        : tC_gAux(cute::forward<GTensor>(tC_gAux)),
          tC_rAux(cute::forward<RTensor>(tC_rAux)),
          tC_cAux(cute::forward<CTensor>(tC_cAux)),
          problem_shape(problem_shape),
          params_ptr(params_ptr) {}

    GTensor tC_gAux;
    RTensor tC_rAux;
    CTensor tC_cAux;
    ProblemShape problem_shape;
    Params const* params_ptr;

    CUTLASS_DEVICE void begin_step(int step_idx) {
      cute::clear(tC_rAux(cute::_, cute::_, cute::_, step_idx % Stages));
      auto dst_v = cute::filter(tC_rAux(cute::_, cute::_, cute::_, step_idx % Stages));
      if (params_ptr->ptr_aux == nullptr) {
        auto dst_elements = cute::recast<Element>(dst_v);
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < cute::size(dst_elements); ++i) dst_elements(i) = params_ptr->null_default;
        return;
      }
      auto src_v = cute::filter(tC_gAux(cute::_, cute::_, cute::_, step_idx));
      auto coord_v = cute::filter(tC_cAux(cute::_, cute::_, cute::_, step_idx));
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < cute::size(src_v); ++i) {
        bool guard = cute::elem_less(coord_v(i), problem_shape);
        cutlass::arch::global_load<VecType, sizeof(VecType)>(dst_v(i), &src_v(i), guard);
      }
    }

    template <class ElementAccumulator_, int FragmentSize>
    CUTLASS_DEVICE auto visit(
        int iter_idx,
        int,
        int,
        int frg_idx,
        cutlass::Array<ElementAccumulator_, FragmentSize> const&) {
      auto fragments = cute::recast<cutlass::Array<Element, FragmentSize>>(
          cute::coalesce(tC_rAux(cute::_, cute::_, cute::_, iter_idx % Stages)));
      return fragments(frg_idx);
    }
  };

  template <class ProblemShape>
  CUTLASS_DEVICE auto get_callbacks(
      cutlass::gemm::GemmCoord threadblock_tile_offset,
      int thread_idx,
      ProblemShape problem_shape) {
    auto mAux = cute::make_tensor(cute::make_gmem_ptr(params_ptr->ptr_aux), problem_shape, params_ptr->dAux);
    auto tC_gAux = cute::recast<VecType>(cute::group_modes<3, 6>(
        ThreadMap::partition(mAux, thread_idx, threadblock_tile_offset)));
    auto tC_rAux = cute::make_tensor<VecType>(cute::make_layout(cute::flatten(cute::make_shape(
        cute::take<0, 3>(tC_gAux.shape()), cute::Int<Stages>{}))));
    auto cAux = cute::make_identity_tensor(mAux.shape());
    auto tC_cAux = cute::outer_partition(
        cute::group_modes<3, 6>(ThreadMap::partition(cAux, thread_idx, threadblock_tile_offset)),
        cute::Shape<cute::Int<VecLength>>{},
        (cute::_0{}));
    return Callbacks<decltype(tC_gAux), decltype(tC_rAux), decltype(tC_cAux), ProblemShape>(
        cute::move(tC_gAux), cute::move(tC_rAux), cute::move(tC_cAux), problem_shape, params_ptr);
  }
};

template <typename ElementOutput, bool kGatedResidual, int kTileM = 128>
struct FusedEpilogueGemm;

template <typename ElementOutput, int kTileM>
struct FusedEpilogueGemm<ElementOutput, false, kTileM> {
  using ThreadblockShape = cutlass::gemm::GemmShape<kTileM, 128, 256>;
  using WarpShape = cutlass::gemm::GemmShape<kTileM == 64 ? 32 : 64, 64, 256>;
  static int const kElementsPerAccess = 8;
  static int const kEpilogueStages = 1;
  using OutputTileThreadMap = cutlass::epilogue::threadblock::OutputTileThreadLayout<
      ThreadblockShape, WarpShape, ElementOutput, kElementsPerAccess, kEpilogueStages>;
  using Accumulator = cutlass::epilogue::threadblock::VisitorAccFetch;
  using ActivationScale = cutlass::epilogue::threadblock::VisitorColBroadcast<OutputTileThreadMap, float>;
  using Multiply = cutlass::epilogue::threadblock::VisitorCompute<
      cutlass::multiplies, float, float, cutlass::FloatRoundStyle::round_to_nearest>;
  using ScaleActivation = cutlass::epilogue::threadblock::Sm80EVT<Multiply, Accumulator, ActivationScale>;
  using WeightScale = cutlass::epilogue::threadblock::VisitorRowBroadcast<
      OutputTileThreadMap, float, cute::Stride<cute::_0, cute::_1, cute::_0>>;
  using ScaleWeight = cutlass::epilogue::threadblock::Sm80EVT<Multiply, ScaleActivation, WeightScale>;
  using Bias = cutlass::epilogue::threadblock::VisitorRowBroadcast<
      OutputTileThreadMap, ElementOutput, cute::Stride<cute::_0, cute::_1, cute::_0>>;
  using Add = cutlass::epilogue::threadblock::VisitorCompute<
      cutlass::plus, float, float, cutlass::FloatRoundStyle::round_to_nearest>;
  using AddBias = cutlass::epilogue::threadblock::Sm80EVT<Add, ScaleWeight, Bias>;
  using Output = cutlass::epilogue::threadblock::VisitorAuxStore<
      OutputTileThreadMap,
      ElementOutput,
      cutlass::FloatRoundStyle::round_to_nearest,
      cute::Stride<int64_t, cute::_1, cute::_0>>;
  using OutputCallbacks = cutlass::epilogue::threadblock::Sm80EVT<Output, AddBias>;
  using Kernel = typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
      ElementA,
      LayoutA,
      cutlass::ComplexTransform::kNone,
      kAlignmentA,
      ElementB,
      LayoutB,
      cutlass::ComplexTransform::kNone,
      kAlignmentB,
      ElementOutput,
      LayoutC,
      kElementsPerAccess,
      ElementAccumulator,
      float,
      cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80,
      ThreadblockShape,
      WarpShape,
      InstructionShape,
      OutputCallbacks,
      ThreadblockSwizzle,
      kMainloopStages,
      cutlass::arch::OpMultiplyAddSaturate,
      kEpilogueStages>::GemmKernel;
  using DeviceGemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
};

template <typename ElementOutput, int kTileM>
struct FusedEpilogueGemm<ElementOutput, true, kTileM> {
  using ThreadblockShape = cutlass::gemm::GemmShape<kTileM, 128, 256>;
  using WarpShape = cutlass::gemm::GemmShape<kTileM == 64 ? 32 : 64, 64, 256>;
  static int const kElementsPerAccess = 8;
  static int const kEpilogueStages = 1;
  using OutputTileThreadMap = cutlass::epilogue::threadblock::OutputTileThreadLayout<
      ThreadblockShape, WarpShape, ElementOutput, kElementsPerAccess, kEpilogueStages>;
  using Accumulator = cutlass::epilogue::threadblock::VisitorAccFetch;
  using ActivationScale = cutlass::epilogue::threadblock::VisitorColBroadcast<OutputTileThreadMap, float>;
  using Multiply = cutlass::epilogue::threadblock::VisitorCompute<
      cutlass::multiplies, float, float, cutlass::FloatRoundStyle::round_to_nearest>;
  using ScaleActivation = cutlass::epilogue::threadblock::Sm80EVT<Multiply, Accumulator, ActivationScale>;
  using WeightScale = cutlass::epilogue::threadblock::VisitorRowBroadcast<
      OutputTileThreadMap, float, cute::Stride<cute::_0, cute::_1, cute::_0>>;
  using ScaleWeight = cutlass::epilogue::threadblock::Sm80EVT<Multiply, ScaleActivation, WeightScale>;
  using Bias = cutlass::epilogue::threadblock::VisitorRowBroadcast<
      OutputTileThreadMap, ElementOutput, cute::Stride<cute::_0, cute::_1, cute::_0>>;
  using Add = cutlass::epilogue::threadblock::VisitorCompute<
      cutlass::plus, float, float, cutlass::FloatRoundStyle::round_to_nearest>;
  using AddBias = cutlass::epilogue::threadblock::Sm80EVT<Add, ScaleWeight, Bias>;
  using Gate = VisitorGroupedRowBroadcast<OutputTileThreadMap, ElementOutput>;
  using ApplyGate = cutlass::epilogue::threadblock::Sm80EVT<Multiply, AddBias, Gate>;
  using Residual = VisitorOptionalAuxLoad<
      OutputTileThreadMap, ElementOutput, cute::Stride<int64_t, cute::_1, cute::_0>>;
  using AddResidual = cutlass::epilogue::threadblock::Sm80EVT<Add, ApplyGate, Residual>;
  using Output = cutlass::epilogue::threadblock::VisitorAuxStore<
      OutputTileThreadMap,
      ElementOutput,
      cutlass::FloatRoundStyle::round_to_nearest,
      cute::Stride<int64_t, cute::_1, cute::_0>>;
  using OutputCallbacks = cutlass::epilogue::threadblock::Sm80EVT<Output, AddResidual>;
  using Kernel = typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
      ElementA,
      LayoutA,
      cutlass::ComplexTransform::kNone,
      kAlignmentA,
      ElementB,
      LayoutB,
      cutlass::ComplexTransform::kNone,
      kAlignmentB,
      ElementOutput,
      LayoutC,
      kElementsPerAccess,
      ElementAccumulator,
      float,
      cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80,
      ThreadblockShape,
      WarpShape,
      InstructionShape,
      OutputCallbacks,
      ThreadblockSwizzle,
      kMainloopStages,
      cutlass::arch::OpMultiplyAddSaturate,
      kEpilogueStages>::GemmKernel;
  using DeviceGemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
};

// This is the power-of-two kernel decomposition used by
// Dao-AILab/fast-hadamard-transform.  The output store is replaced by a
// per-token S4 quantizer, so the rotated BF16/FP16 tensor never reaches global
// memory between the transform and the CUTLASS GEMM.
template <int kNThreads_, int kLogN_, typename input_t_>
struct FusedRHTTraits {
  using input_t = input_t_;
  static constexpr int kNThreads = kNThreads_;
  static constexpr int kLogN = kLogN_;
  static constexpr int N = 1 << kLogN;
  static constexpr int kNBytes = sizeof(input_t);
  static constexpr int kNElts = kNBytes == 4 ? 4 : 8;
  static constexpr int kNExchangePerVec = sizeof(float) / sizeof(input_t);
  using vec_t = typename BytesToType<kNBytes * kNElts>::Type;
  static constexpr int kNChunks = N / (kNElts * kNThreads);
  static constexpr int kSmemExchangeSize = std::min(N * 4, 32 * 1024);
  static constexpr int kNExchangeRounds = N * 4 / kSmemExchangeSize;
  static constexpr int kSmemSize = kSmemExchangeSize;
  static_assert(kNBytes == 2, "The fused RHT path expects FP16 or BF16 input");
  static_assert(kNChunks > 0 && kNChunks * kNElts * kNThreads == N);
  static_assert(kNExchangeRounds * kSmemExchangeSize == N * 4);
};

template <typename scalar_t>
__device__ __forceinline__ float as_float(scalar_t value) {
  return static_cast<float>(value);
}

__device__ __forceinline__ float warp_max(float value) {
#pragma unroll
  for (int offset = 16; offset; offset >>= 1) {
    value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset));
  }
  return value;
}

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
  for (int offset = 16; offset; offset >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return value;
}

template <typename scalar_t, bool kInputScaleIsInverse>
__device__ __forceinline__ float transform_value(
    scalar_t value,
    float const* input_scale,
    int64_t column) {
  float transformed = as_float(value);
  if (input_scale != nullptr) {
    if constexpr (kInputScaleIsInverse) transformed *= input_scale[column];
    else transformed /= input_scale[column];
  }
  return transformed;
}

template <typename scalar_t>
__device__ __forceinline__ float adaln_value(
    scalar_t value,
    float mean,
    float inv_std,
    scalar_t const* scale,
    scalar_t const* shift,
    int64_t modulation_offset,
    int64_t column) {
  float normalized = (as_float(value) - mean) * inv_std;
  return normalized * (1.0f + as_float(scale[modulation_offset + column])) +
      as_float(shift[modulation_offset + column]);
}

template <int kNChunks, int kNElts, typename input_t>
__device__ __forceinline__ void load_signed_input(
    input_t const* input,
    float const* signs,
    float values[kNChunks][kNElts]) {
#pragma unroll
  for (int chunk = 0; chunk < kNChunks; ++chunk) {
    int base = (chunk * blockDim.x + threadIdx.x) * kNElts;
#pragma unroll
    for (int item = 0; item < kNElts; ++item) {
      int index = base + item;
      values[chunk][item] = static_cast<float>(input[index]) * signs[index];
    }
  }
}

template <typename Ktraits, bool kStaticWam, bool kAdaLN>
__global__ __launch_bounds__(Ktraits::kNThreads) void fused_rht_quantize_s4_kernel(
    typename Ktraits::input_t const* __restrict__ input,
    float const* __restrict__ signs,
    float const* __restrict__ input_scale_inverse,
    typename Ktraits::input_t const* __restrict__ adaln_scale,
    typename Ktraits::input_t const* __restrict__ adaln_shift,
    int64_t modulation_row_stride,
    int64_t modulation_span,
    float epsilon,
    float const* __restrict__ stream_gains,
    float const* __restrict__ activation_scale,
    uint8_t* __restrict__ output,
    float* __restrict__ scales,
    int64_t stream_span,
    int64_t num_streams,
    float clip_ratio) {
  static_assert(!kAdaLN || kStaticWam, "AdaLN RHT is only used by the static WAM producer");
  constexpr int kNThreads = Ktraits::kNThreads;
  constexpr int kNElts = Ktraits::kNElts;
  constexpr int kNExchangePerVec = Ktraits::kNExchangePerVec;
  constexpr int kNChunks = Ktraits::kNChunks;
  using input_t = typename Ktraits::input_t;
  using vec_t = typename Ktraits::vec_t;

  constexpr int kLogNElts = cilog2(kNElts);
  constexpr int kWarpSize = std::min(kNThreads, 32);
  constexpr int kLogWarpSize = cilog2(kWarpSize);
  constexpr int kNWarps = kNThreads / kWarpSize;
  constexpr int kLogNWarps = cilog2(kNWarps);
  constexpr int kLoadsPerExchange = Ktraits::kSmemExchangeSize / (sizeof(vec_t) * kNThreads);
  constexpr int kChunksPerExchange =
      Ktraits::kSmemExchangeSize / (sizeof(vec_t) * kNExchangePerVec * kNThreads);
  static_assert(kLoadsPerExchange * sizeof(vec_t) * kNThreads == Ktraits::kSmemExchangeSize);
  static_assert(
      Ktraits::kNExchangeRounds * kLoadsPerExchange * sizeof(vec_t) ==
      kNChunks * kNElts * sizeof(float));
  static_assert(kChunksPerExchange > 0 && kNChunks % kChunksPerExchange == 0);

  extern __shared__ char dynamic_smem[];
  vec_t* smem_exchange = reinterpret_cast<vec_t*>(dynamic_smem);
  __shared__ float warp_values[8];
  __shared__ float warp_squares[8];
  __shared__ float mean;
  __shared__ float inv_std;
  __shared__ float raw_scale;

  int row = blockIdx.x;
  input_t const* row_input = input + static_cast<int64_t>(row) * Ktraits::N;
  if constexpr (kAdaLN) {
    float sum = 0.0f;
    float square_sum = 0.0f;
    for (int column = threadIdx.x; column < Ktraits::N; column += blockDim.x) {
      float value = static_cast<float>(row_input[column]);
      sum += value;
      square_sum += value * value;
    }
    sum = warp_sum(sum);
    square_sum = warp_sum(square_sum);
    if ((threadIdx.x & 31) == 0) {
      warp_values[threadIdx.x >> 5] = sum;
      warp_squares[threadIdx.x >> 5] = square_sum;
    }
    __syncthreads();
    float block_sum = threadIdx.x < kNWarps ? warp_values[threadIdx.x] : 0.0f;
    float block_square_sum = threadIdx.x < kNWarps ? warp_squares[threadIdx.x] : 0.0f;
    if (threadIdx.x < 32) {
      block_sum = warp_sum(block_sum);
      block_square_sum = warp_sum(block_square_sum);
    }
    if (threadIdx.x == 0) {
      float inverse_columns = 1.0f / static_cast<float>(Ktraits::N);
      mean = block_sum * inverse_columns;
      float variance = fmaxf(block_square_sum * inverse_columns - mean * mean, 0.0f);
      inv_std = rsqrtf(variance + epsilon);
    }
    __syncthreads();
  }

  float values[kNChunks][kNElts];
  int64_t modulation_offset = kAdaLN
      ? (static_cast<int64_t>(row) / modulation_span) * modulation_row_stride
      : 0;
#pragma unroll
  for (int chunk = 0; chunk < kNChunks; ++chunk) {
    int base = (chunk * blockDim.x + threadIdx.x) * kNElts;
#pragma unroll
    for (int item = 0; item < kNElts; ++item) {
      int index = base + item;
      float value = static_cast<float>(row_input[index]);
      if constexpr (kAdaLN) {
        value = (value - mean) * inv_std;
        value = value * (1.0f + static_cast<float>(adaln_scale[modulation_offset + index])) +
            static_cast<float>(adaln_shift[modulation_offset + index]);
      }
      values[chunk][item] = value * signs[index];
    }
  }

  hadamard_mult_thread<kLogNElts, kNChunks>(values);
  hadamard_mult_warp<kLogWarpSize, 0, kNChunks, kNElts>(values);

  if constexpr (kNWarps > 1) {
    exchange_smem_pre<kNChunks, kChunksPerExchange, kNElts, kWarpSize, kNWarps, true, vec_t>(
        values, smem_exchange);
    hadamard_mult_warp<kLogNWarps, 0, kNChunks, kNElts>(values);
    exchange_smem_pre<kNChunks, kChunksPerExchange, kNElts, kWarpSize, kNWarps, false, vec_t>(
        values, smem_exchange);
  }

  if constexpr (kNChunks > 1) {
    float transposed[kNElts][kNChunks];
#pragma unroll
    for (int chunk = 0; chunk < kNChunks; ++chunk) {
#pragma unroll
      for (int item = 0; item < kNElts; ++item) transposed[item][chunk] = values[chunk][item];
    }
    constexpr int kLogNChunks = cilog2(kNChunks);
    hadamard_mult_thread<kLogNChunks, kNElts>(transposed);
#pragma unroll
    for (int chunk = 0; chunk < kNChunks; ++chunk) {
#pragma unroll
      for (int item = 0; item < kNElts; ++item) values[chunk][item] = transposed[item][chunk];
    }
  }

  float maximum = 0.0f;
#pragma unroll
  for (int chunk = 0; chunk < kNChunks; ++chunk) {
#pragma unroll
    for (int item = 0; item < kNElts; ++item) maximum = fmaxf(maximum, fabsf(values[chunk][item]));
  }
  if constexpr (kStaticWam) {
    if (threadIdx.x == 0) {
      int64_t stream = (static_cast<int64_t>(row) / stream_span) % num_streams;
      float scale = activation_scale[0];
      float gain = stream_gains[stream];
      // The Hadamard implementation holds the unnormalized transform in
      // registers. Fold 1/sqrt(K), WAM's stream gain and the static A4 scale
      // into one quantization multiplier, while retaining scale/gain for EVT.
      raw_scale = rsqrtf(static_cast<float>(Ktraits::N)) * gain / scale;
      scales[row] = scale / gain;
    }
  } else {
    maximum = warp_max(maximum);
    if ((threadIdx.x & 31) == 0) warp_values[threadIdx.x >> 5] = maximum;
    __syncthreads();
    maximum = threadIdx.x < kNWarps ? warp_values[threadIdx.x] : 0.0f;
    if (threadIdx.x < 32) maximum = warp_max(maximum);
    if (threadIdx.x == 0) {
      raw_scale = fmaxf(maximum * clip_ratio, 1.0e-8f) / 7.0f;
      scales[row] = raw_scale * rsqrtf(static_cast<float>(Ktraits::N));
    }
  }
  __syncthreads();

  float quant_multiplier = kStaticWam ? raw_scale : 1.0f / raw_scale;
  uint8_t* row_output = output + static_cast<int64_t>(row) * (Ktraits::N / 2);
#pragma unroll
  for (int chunk = 0; chunk < kNChunks; ++chunk) {
    int packed_base = (chunk * blockDim.x + threadIdx.x) * (kNElts / 2);
#pragma unroll
    for (int item = 0; item < kNElts; item += 2) {
      int first_index = (chunk * blockDim.x + threadIdx.x) * kNElts + item;
      float first_multiplier = quant_multiplier;
      float second_multiplier = quant_multiplier;
      if constexpr (kStaticWam) {
        first_multiplier *= input_scale_inverse[first_index];
        second_multiplier *= input_scale_inverse[first_index + 1];
      }
      int first = __float2int_rn(values[chunk][item] * first_multiplier);
      int second = __float2int_rn(values[chunk][item + 1] * second_multiplier);
      first = max(-7, min(7, first));
      second = max(-7, min(7, second));
      row_output[packed_base + item / 2] =
          static_cast<uint8_t>((first & 0x0f) | ((second & 0x0f) << 4));
    }
  }
}

template <bool kStaticWam, bool kAdaLN, int kNThreads, int kLogN, typename input_t>
void launch_fused_rht_quantize(
    input_t const* input,
    float const* signs,
    float const* input_scale_inverse,
    input_t const* adaln_scale,
    input_t const* adaln_shift,
    int64_t modulation_row_stride,
    int64_t modulation_span,
    float epsilon,
    float const* stream_gains,
    float const* activation_scale,
    uint8_t* output,
    float* scales,
    int64_t rows,
    int64_t stream_span,
    int64_t num_streams,
    float clip_ratio,
    cudaStream_t stream) {
  using Ktraits = FusedRHTTraits<kNThreads, kLogN, input_t>;
  auto kernel = &fused_rht_quantize_s4_kernel<Ktraits, kStaticWam, kAdaLN>;
  kernel<<<rows, kNThreads, Ktraits::kSmemSize, stream>>>(
      input, signs, input_scale_inverse, adaln_scale, adaln_shift,
      modulation_row_stride, modulation_span, epsilon, stream_gains, activation_scale,
      output, scales, stream_span, num_streams, clip_ratio);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// FastWAMJoint contains non-power-of-two projection widths (3072 and 14336).
// Its reference RHT definition independently transforms the largest
// power-of-two blocks that tile K, but computes one activation scale across the
// complete token. Keep every transformed block in registers until that shared
// reduction is known, then pack the full token without a global FP intermediate.
template <typename Ktraits, int kBlocks>
__global__ __launch_bounds__(Ktraits::kNThreads) void fused_rht_block_quantize_s4_kernel(
    typename Ktraits::input_t const* __restrict__ input,
    float const* __restrict__ signs,
    uint8_t* __restrict__ output,
    float* __restrict__ scales,
    float clip_ratio) {
  constexpr int kNThreads = Ktraits::kNThreads;
  constexpr int kNElts = Ktraits::kNElts;
  constexpr int kNExchangePerVec = Ktraits::kNExchangePerVec;
  constexpr int kNChunks = Ktraits::kNChunks;
  constexpr int kBlockN = Ktraits::N;
  constexpr int kTotalN = kBlocks * kBlockN;
  using input_t = typename Ktraits::input_t;
  using vec_t = typename Ktraits::vec_t;

  constexpr int kLogNElts = cilog2(kNElts);
  constexpr int kWarpSize = std::min(kNThreads, 32);
  constexpr int kLogWarpSize = cilog2(kWarpSize);
  constexpr int kNWarps = kNThreads / kWarpSize;
  constexpr int kLogNWarps = cilog2(kNWarps);
  constexpr int kLoadsPerExchange = Ktraits::kSmemExchangeSize / (sizeof(vec_t) * kNThreads);
  constexpr int kChunksPerExchange =
      Ktraits::kSmemExchangeSize / (sizeof(vec_t) * kNExchangePerVec * kNThreads);
  static_assert(kLoadsPerExchange * sizeof(vec_t) * kNThreads == Ktraits::kSmemExchangeSize);
  static_assert(
      Ktraits::kNExchangeRounds * kLoadsPerExchange * sizeof(vec_t) ==
      kNChunks * kNElts * sizeof(float));
  static_assert(kChunksPerExchange > 0 && kNChunks % kChunksPerExchange == 0);

  extern __shared__ char dynamic_smem[];
  vec_t* smem_exchange = reinterpret_cast<vec_t*>(dynamic_smem);
  __shared__ float warp_values[8];
  __shared__ float raw_scale;

  int row = blockIdx.x;
  input_t const* row_input = input + static_cast<int64_t>(row) * kTotalN;
  float values[kBlocks][kNChunks][kNElts];

#pragma unroll
  for (int block = 0; block < kBlocks; ++block) {
#pragma unroll
    for (int chunk = 0; chunk < kNChunks; ++chunk) {
      int base = (chunk * blockDim.x + threadIdx.x) * kNElts;
#pragma unroll
      for (int item = 0; item < kNElts; ++item) {
        int index = block * kBlockN + base + item;
        values[block][chunk][item] = static_cast<float>(row_input[index]) * signs[index];
      }
    }

    hadamard_mult_thread<kLogNElts, kNChunks>(values[block]);
    hadamard_mult_warp<kLogWarpSize, 0, kNChunks, kNElts>(values[block]);

    if constexpr (kNWarps > 1) {
      exchange_smem_pre<kNChunks, kChunksPerExchange, kNElts, kWarpSize, kNWarps, true, vec_t>(
          values[block], smem_exchange);
      hadamard_mult_warp<kLogNWarps, 0, kNChunks, kNElts>(values[block]);
      exchange_smem_pre<kNChunks, kChunksPerExchange, kNElts, kWarpSize, kNWarps, false, vec_t>(
          values[block], smem_exchange);
    }

    if constexpr (kNChunks > 1) {
      float transposed[kNElts][kNChunks];
#pragma unroll
      for (int chunk = 0; chunk < kNChunks; ++chunk) {
#pragma unroll
        for (int item = 0; item < kNElts; ++item) {
          transposed[item][chunk] = values[block][chunk][item];
        }
      }
      constexpr int kLogNChunks = cilog2(kNChunks);
      hadamard_mult_thread<kLogNChunks, kNElts>(transposed);
#pragma unroll
      for (int chunk = 0; chunk < kNChunks; ++chunk) {
#pragma unroll
        for (int item = 0; item < kNElts; ++item) {
          values[block][chunk][item] = transposed[item][chunk];
        }
      }
    }
  }

  float maximum = 0.0f;
#pragma unroll
  for (int block = 0; block < kBlocks; ++block) {
#pragma unroll
    for (int chunk = 0; chunk < kNChunks; ++chunk) {
#pragma unroll
      for (int item = 0; item < kNElts; ++item) {
        maximum = fmaxf(maximum, fabsf(values[block][chunk][item]));
      }
    }
  }
  maximum = warp_max(maximum);
  if ((threadIdx.x & 31) == 0) warp_values[threadIdx.x >> 5] = maximum;
  __syncthreads();
  maximum = threadIdx.x < kNWarps ? warp_values[threadIdx.x] : 0.0f;
  if (threadIdx.x < 32) maximum = warp_max(maximum);
  if (threadIdx.x == 0) {
    raw_scale = fmaxf(maximum * clip_ratio, 1.0e-8f) / 7.0f;
    scales[row] = raw_scale * rsqrtf(static_cast<float>(kBlockN));
  }
  __syncthreads();

  float quant_multiplier = 1.0f / raw_scale;
  uint8_t* row_output = output + static_cast<int64_t>(row) * (kTotalN / 2);
#pragma unroll
  for (int block = 0; block < kBlocks; ++block) {
#pragma unroll
    for (int chunk = 0; chunk < kNChunks; ++chunk) {
      int packed_base = block * (kBlockN / 2) +
          (chunk * blockDim.x + threadIdx.x) * (kNElts / 2);
#pragma unroll
      for (int item = 0; item < kNElts; item += 2) {
        int first = __float2int_rn(values[block][chunk][item] * quant_multiplier);
        int second = __float2int_rn(values[block][chunk][item + 1] * quant_multiplier);
        first = max(-7, min(7, first));
        second = max(-7, min(7, second));
        row_output[packed_base + item / 2] =
            static_cast<uint8_t>((first & 0x0f) | ((second & 0x0f) << 4));
      }
    }
  }
}

template <int kNThreads, int kLogBlockN, int kBlocks, typename input_t>
void launch_fused_rht_block_quantize(
    input_t const* input,
    float const* signs,
    uint8_t* output,
    float* scales,
    int64_t rows,
    float clip_ratio,
    cudaStream_t stream) {
  using Ktraits = FusedRHTTraits<kNThreads, kLogBlockN, input_t>;
  auto kernel = &fused_rht_block_quantize_s4_kernel<Ktraits, kBlocks>;
  kernel<<<rows, kNThreads, Ktraits::kSmemSize, stream>>>(
      input, signs, output, scales, clip_ratio);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename input_t>
void dispatch_fused_rht_quantize(
    input_t const* input,
    float const* signs,
    uint8_t* output,
    float* scales,
    int64_t rows,
    int64_t columns,
    float clip_ratio,
    cudaStream_t stream) {
#define LAUNCH_DYNAMIC_RHT(threads, log_n) \
  launch_fused_rht_quantize<false, false, threads, log_n>( \
      input, signs, nullptr, static_cast<input_t const*>(nullptr), static_cast<input_t const*>(nullptr), \
      0, 1, 0.0f, nullptr, nullptr, \
      output, scales, rows, 1, 1, clip_ratio, stream)
  switch (columns) {
    case 256:
      LAUNCH_DYNAMIC_RHT(32, 8);
      break;
    case 512:
      LAUNCH_DYNAMIC_RHT(32, 9);
      break;
    case 1024:
      LAUNCH_DYNAMIC_RHT(128, 10);
      break;
    case 2048:
      LAUNCH_DYNAMIC_RHT(256, 11);
      break;
    case 4096:
      LAUNCH_DYNAMIC_RHT(256, 12);
      break;
    case 8192:
      LAUNCH_DYNAMIC_RHT(256, 13);
      break;
    case 3072:
      launch_fused_rht_block_quantize<128, 10, 3>(
          input, signs, output, scales, rows, clip_ratio, stream);
      break;
    case 14336:
      launch_fused_rht_block_quantize<256, 11, 7>(
          input, signs, output, scales, rows, clip_ratio, stream);
      break;
    default:
      TORCH_CHECK(
          false,
          "Fused RHT supports power-of-two K from 256 through 8192 and "
          "FastWAMJoint block widths 3072/14336; got ",
          columns);
  }
#undef LAUNCH_DYNAMIC_RHT
}

#include "rht.cuh"

template <bool kAdaLN, typename input_t>
void dispatch_fused_rht_wam_quantize(
    input_t const* input,
    float const* signs,
    float const* input_scale_inverse,
    input_t const* adaln_scale,
    input_t const* adaln_shift,
    int64_t modulation_row_stride,
    int64_t modulation_span,
    float epsilon,
    float const* stream_gains,
    float const* activation_scale,
    uint8_t* output,
    float* scales,
    int64_t rows,
    int64_t columns,
    int64_t stream_span,
    int64_t num_streams,
    cudaStream_t stream) {
#define LAUNCH_STATIC_RHT(threads, log_n) \
  launch_fused_rht_quantize<true, kAdaLN, threads, log_n>( \
      input, signs, input_scale_inverse, adaln_scale, adaln_shift, \
      modulation_row_stride, modulation_span, epsilon, stream_gains, activation_scale, \
      output, scales, rows, stream_span, num_streams, 1.0f, stream)
  switch (columns) {
    case 256:
      LAUNCH_STATIC_RHT(32, 8);
      break;
    case 512:
      LAUNCH_STATIC_RHT(32, 9);
      break;
    case 1024:
      LAUNCH_STATIC_RHT(128, 10);
      break;
    case 2048:
      LAUNCH_STATIC_RHT(256, 11);
      break;
    case 4096:
      LAUNCH_STATIC_RHT(256, 12);
      break;
    case 8192:
      LAUNCH_STATIC_RHT(256, 13);
      break;
    default:
      TORCH_CHECK(false, "Fused RHT WAM supports power-of-two K from 256 through 8192; got ", columns);
  }
#undef LAUNCH_STATIC_RHT
}

template <typename scalar_t, bool kInputScaleIsInverse>
__global__ void quantize_per_token_s4_kernel(
    scalar_t const* __restrict__ input,
    float const* __restrict__ input_scale,
    uint8_t* __restrict__ output,
    float* __restrict__ scales,
    int64_t rows,
    int64_t columns,
    float clip_ratio) {
  int64_t row = blockIdx.x;
  if (row >= rows) return;
  __shared__ float warp_values[16];
  __shared__ float row_scale;

  float maximum = 0.0f;
  for (int64_t column = threadIdx.x; column < columns; column += blockDim.x) {
    maximum = fmaxf(maximum, fabsf(transform_value<scalar_t, kInputScaleIsInverse>(
        input[row * columns + column], input_scale, column)));
  }
  maximum = warp_max(maximum);
  if ((threadIdx.x & 31) == 0) warp_values[threadIdx.x >> 5] = maximum;
  __syncthreads();
  maximum = threadIdx.x < blockDim.x / 32 ? warp_values[threadIdx.x] : 0.0f;
  if (threadIdx.x < 32) maximum = warp_max(maximum);
  if (threadIdx.x == 0) {
    row_scale = fmaxf(maximum * clip_ratio, 1.0e-8f) / 7.0f;
    scales[row] = row_scale;
  }
  __syncthreads();

  float inverse = 1.0f / row_scale;
  int64_t packed_columns = columns / 2;
  for (int64_t packed_column = threadIdx.x; packed_column < packed_columns; packed_column += blockDim.x) {
    int64_t column = packed_column * 2;
    int first = __float2int_rn(transform_value<scalar_t, kInputScaleIsInverse>(
        input[row * columns + column], input_scale, column) * inverse);
    int second = __float2int_rn(transform_value<scalar_t, kInputScaleIsInverse>(
        input[row * columns + column + 1], input_scale, column + 1) * inverse);
    first = max(-7, min(7, first));
    second = max(-7, min(7, second));
    output[row * packed_columns + packed_column] =
        static_cast<uint8_t>((first & 0x0f) | ((second & 0x0f) << 4));
  }
}

template <typename scalar_t, bool kInputScaleIsInverse>
__global__ void quantize_dynamic_adaln_s4_kernel(
    scalar_t const* __restrict__ input,
    scalar_t const* __restrict__ adaln_scale,
    scalar_t const* __restrict__ adaln_shift,
    int64_t modulation_row_stride,
    int64_t modulation_span,
    float epsilon,
    float const* __restrict__ input_scale,
    uint8_t* __restrict__ output,
    float* __restrict__ scales,
    int64_t rows,
    int64_t columns) {
  int64_t row = blockIdx.x;
  if (row >= rows) return;
  __shared__ float warp_values[16];
  __shared__ float warp_squares[16];
  __shared__ float mean;
  __shared__ float inv_std;
  __shared__ float row_scale;
  extern __shared__ float transformed[];

  float sum = 0.0f;
  float square_sum = 0.0f;
  for (int64_t column = threadIdx.x; column < columns; column += blockDim.x) {
    float value = as_float(input[row * columns + column]);
    sum += value;
    square_sum += value * value;
  }
  sum = warp_sum(sum);
  square_sum = warp_sum(square_sum);
  if ((threadIdx.x & 31) == 0) {
    warp_values[threadIdx.x >> 5] = sum;
    warp_squares[threadIdx.x >> 5] = square_sum;
  }
  __syncthreads();
  float block_sum = threadIdx.x < blockDim.x / 32 ? warp_values[threadIdx.x] : 0.0f;
  float block_square_sum = threadIdx.x < blockDim.x / 32 ? warp_squares[threadIdx.x] : 0.0f;
  if (threadIdx.x < 32) {
    block_sum = warp_sum(block_sum);
    block_square_sum = warp_sum(block_square_sum);
  }
  if (threadIdx.x == 0) {
    float inverse_columns = 1.0f / static_cast<float>(columns);
    mean = block_sum * inverse_columns;
    float variance = fmaxf(block_square_sum * inverse_columns - mean * mean, 0.0f);
    inv_std = rsqrtf(variance + epsilon);
  }
  __syncthreads();

  int64_t modulation_row = row / modulation_span;
  int64_t modulation_offset = modulation_row * modulation_row_stride;
  float maximum = 0.0f;
  for (int64_t column = threadIdx.x; column < columns; column += blockDim.x) {
    float value = adaln_value(
        input[row * columns + column], mean, inv_std,
        adaln_scale, adaln_shift, modulation_offset, column);
    if (input_scale != nullptr) {
      if constexpr (kInputScaleIsInverse) value *= input_scale[column];
      else value /= input_scale[column];
    }
    transformed[column] = value;
    maximum = fmaxf(maximum, fabsf(value));
  }
  maximum = warp_max(maximum);
  if ((threadIdx.x & 31) == 0) warp_values[threadIdx.x >> 5] = maximum;
  __syncthreads();
  maximum = threadIdx.x < blockDim.x / 32 ? warp_values[threadIdx.x] : 0.0f;
  if (threadIdx.x < 32) maximum = warp_max(maximum);
  if (threadIdx.x == 0) {
    row_scale = fmaxf(maximum, 1.0e-8f) / 7.0f;
    scales[row] = row_scale;
  }
  __syncthreads();

  float inverse = 1.0f / row_scale;
  int64_t packed_columns = columns / 2;
  for (int64_t packed_column = threadIdx.x; packed_column < packed_columns; packed_column += blockDim.x) {
    int64_t column = packed_column * 2;
    int first = __float2int_rn(transformed[column] * inverse);
    int second = __float2int_rn(transformed[column + 1] * inverse);
    first = max(-7, min(7, first));
    second = max(-7, min(7, second));
    output[row * packed_columns + packed_column] =
        static_cast<uint8_t>((first & 0x0f) | ((second & 0x0f) << 4));
  }
}

template <typename scalar_t, bool kInputScaleIsInverse>
__global__ void quantize_static_s4_kernel(
    scalar_t const* __restrict__ input,
    float const* __restrict__ input_scale,
    scalar_t const* __restrict__ row_gains,
    float const* __restrict__ activation_scale,
    uint8_t* __restrict__ output,
    float* __restrict__ scales,
    int64_t rows,
    int64_t columns) {
  int64_t row = blockIdx.x;
  if (row >= rows) return;
  __shared__ float gain_over_scale;
  if (threadIdx.x == 0) {
    float scale = activation_scale[0];
    float gain = as_float(row_gains[row]);
    gain_over_scale = gain / scale;
    scales[row] = scale / gain;
  }
  __syncthreads();
  int64_t packed_columns = columns / 2;
  for (int64_t packed_column = threadIdx.x; packed_column < packed_columns; packed_column += blockDim.x) {
    int64_t column = packed_column * 2;
    int first = __float2int_rn(transform_value<scalar_t, kInputScaleIsInverse>(
        input[row * columns + column], input_scale, column) * gain_over_scale);
    int second = __float2int_rn(transform_value<scalar_t, kInputScaleIsInverse>(
        input[row * columns + column + 1], input_scale, column + 1) * gain_over_scale);
    first = max(-7, min(7, first));
    second = max(-7, min(7, second));
    output[row * packed_columns + packed_column] =
        static_cast<uint8_t>((first & 0x0f) | ((second & 0x0f) << 4));
  }
}

template <typename scalar_t, bool kInputScaleIsInverse>
__global__ void quantize_static_stream_s4_kernel(
    scalar_t const* __restrict__ input,
    float const* __restrict__ input_scale,
    float const* __restrict__ stream_gains,
    float const* __restrict__ activation_scale,
    uint8_t* __restrict__ output,
    float* __restrict__ scales,
    int64_t rows,
    int64_t columns,
    int64_t stream_span,
    int64_t num_streams,
    int64_t sequence_tokens) {
  int64_t row = blockIdx.x;
  if (row >= rows) return;
  __shared__ float gain_over_scale;
  if (threadIdx.x == 0) {
    // Negative span encodes the proprio suffix; restart at each batch item.
    int64_t stream = stream_span < 0
        ? ((row % sequence_tokens) >= sequence_tokens + stream_span ? 1 : 0)
        : (row / stream_span) % num_streams;
    float scale = activation_scale[0];
    float gain = stream_gains[stream];
    gain_over_scale = gain / scale;
    scales[row] = scale / gain;
  }
  __syncthreads();
  int64_t packed_columns = columns / 2;
  for (int64_t packed_column = threadIdx.x; packed_column < packed_columns; packed_column += blockDim.x) {
    int64_t column = packed_column * 2;
    int first = __float2int_rn(transform_value<scalar_t, kInputScaleIsInverse>(
        input[row * columns + column], input_scale, column) * gain_over_scale);
    int second = __float2int_rn(transform_value<scalar_t, kInputScaleIsInverse>(
        input[row * columns + column + 1], input_scale, column + 1) * gain_over_scale);
    first = max(-7, min(7, first));
    second = max(-7, min(7, second));
    output[row * packed_columns + packed_column] =
        static_cast<uint8_t>((first & 0x0f) | ((second & 0x0f) << 4));
  }
}

template <typename scalar_t, bool kInputScaleIsInverse>
__global__ void quantize_static_adaln_stream_s4_kernel(
    scalar_t const* __restrict__ input,
    scalar_t const* __restrict__ adaln_scale,
    scalar_t const* __restrict__ adaln_shift,
    int64_t modulation_row_stride,
    int64_t modulation_span,
    float epsilon,
    float const* __restrict__ input_scale,
    float const* __restrict__ stream_gains,
    float const* __restrict__ activation_scale,
    uint8_t* __restrict__ output,
    float* __restrict__ scales,
    int64_t rows,
    int64_t columns,
    int64_t stream_span,
    int64_t num_streams) {
  int64_t row = blockIdx.x;
  if (row >= rows) return;
  __shared__ float warp_values[16];
  __shared__ float warp_squares[16];
  __shared__ float mean;
  __shared__ float inv_std;
  __shared__ float gain_over_scale;

  float sum = 0.0f;
  float square_sum = 0.0f;
  for (int64_t column = threadIdx.x; column < columns; column += blockDim.x) {
    float value = as_float(input[row * columns + column]);
    sum += value;
    square_sum += value * value;
  }
  sum = warp_sum(sum);
  square_sum = warp_sum(square_sum);
  if ((threadIdx.x & 31) == 0) {
    warp_values[threadIdx.x >> 5] = sum;
    warp_squares[threadIdx.x >> 5] = square_sum;
  }
  __syncthreads();
  float block_sum = threadIdx.x < blockDim.x / 32 ? warp_values[threadIdx.x] : 0.0f;
  float block_square_sum = threadIdx.x < blockDim.x / 32 ? warp_squares[threadIdx.x] : 0.0f;
  if (threadIdx.x < 32) {
    block_sum = warp_sum(block_sum);
    block_square_sum = warp_sum(block_square_sum);
  }
  if (threadIdx.x == 0) {
    float inverse_columns = 1.0f / static_cast<float>(columns);
    mean = block_sum * inverse_columns;
    float variance = fmaxf(block_square_sum * inverse_columns - mean * mean, 0.0f);
    inv_std = rsqrtf(variance + epsilon);
    int64_t stream = (row / stream_span) % num_streams;
    float scale = activation_scale[0];
    float gain = stream_gains[stream];
    gain_over_scale = gain / scale;
    scales[row] = scale / gain;
  }
  __syncthreads();

  int64_t modulation_row = row / modulation_span;
  int64_t modulation_offset = modulation_row * modulation_row_stride;
  int64_t packed_columns = columns / 2;
  for (int64_t packed_column = threadIdx.x; packed_column < packed_columns; packed_column += blockDim.x) {
    int64_t column = packed_column * 2;
    float first_value = adaln_value(
        input[row * columns + column], mean, inv_std,
        adaln_scale, adaln_shift, modulation_offset, column);
    float second_value = adaln_value(
        input[row * columns + column + 1], mean, inv_std,
        adaln_scale, adaln_shift, modulation_offset, column + 1);
    if (input_scale != nullptr) {
      if constexpr (kInputScaleIsInverse) {
        first_value *= input_scale[column];
        second_value *= input_scale[column + 1];
      } else {
        first_value /= input_scale[column];
        second_value /= input_scale[column + 1];
      }
    }
    int first = __float2int_rn(first_value * gain_over_scale);
    int second = __float2int_rn(second_value * gain_over_scale);
    first = max(-7, min(7, first));
    second = max(-7, min(7, second));
    output[row * packed_columns + packed_column] =
        static_cast<uint8_t>((first & 0x0f) | ((second & 0x0f) << 4));
  }
}

void check_cutlass(cutlass::Status status, char const* operation) {
  TORCH_CHECK(status == cutlass::Status::kSuccess, operation, " failed: ", cutlassGetStatusString(status));
}

template <typename Gemm>
void launch_fused_w4a4_gemm_impl(
    torch::Tensor packed_activation,
    torch::Tensor packed_weight,
    typename Gemm::OutputCallbacks::Arguments const& output_callbacks,
    cudaStream_t stream) {
  using DeviceGemm = typename Gemm::DeviceGemm;
  int m = static_cast<int>(packed_activation.size(0));
  int k = static_cast<int>(packed_activation.size(1) * 2);
  int n = static_cast<int>(packed_weight.size(0));
  cutlass::gemm::GemmCoord problem_size(m, n, k);
  typename DeviceGemm::Arguments arguments(
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem_size,
      1,
      output_callbacks,
      reinterpret_cast<ElementA const*>(packed_activation.data_ptr<uint8_t>()),
      reinterpret_cast<ElementB const*>(packed_weight.data_ptr<uint8_t>()),
      nullptr,
      nullptr,
      0,
      0,
      0,
      0,
      k,
      k,
      0,
      0);
  DeviceGemm gemm;
  check_cutlass(gemm.can_implement(arguments), "CUTLASS fused W4A4 can_implement");
  check_cutlass(gemm(arguments, nullptr, stream), "CUTLASS fused W4A4 launch");
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename ElementOutput, int kTileM>
void launch_fused_w4a4_gemm_tiled(
    torch::Tensor packed_activation,
    torch::Tensor packed_weight,
    torch::Tensor activation_scales,
    torch::Tensor weight_scales,
    std::optional<torch::Tensor> const& bias,
    std::optional<torch::Tensor> const& residual,
    std::optional<torch::Tensor> const& gate,
    int64_t gate_span,
    int64_t gate_row_stride,
    torch::Tensor output,
    cudaStream_t stream) {
  int64_t n = packed_weight.size(0);
  if (!residual.has_value()) {
    using Gemm = FusedEpilogueGemm<ElementOutput, false, kTileM>;
    typename Gemm::OutputCallbacks::Arguments callbacks{
        {
            {
                {
                    {},
                    {activation_scales.data_ptr<float>(), 0.0f, {}},
                    {},
                },
                {weight_scales.data_ptr<float>(), 0.0f, {}},
                {},
            },
            {
                bias.has_value() ? reinterpret_cast<ElementOutput const*>(bias->data_ptr()) : nullptr,
                ElementOutput(0),
                {},
            },
            {},
        },
        {
            reinterpret_cast<ElementOutput*>(output.data_ptr()),
            {int64_t(n), cute::_1{}, cute::_0{}},
        },
    };
    launch_fused_w4a4_gemm_impl<Gemm>(packed_activation, packed_weight, callbacks, stream);
    return;
  }

  TORCH_CHECK(gate.has_value(), "fused residual W4A4 GEMM requires a gate tensor");
  using Gemm = FusedEpilogueGemm<ElementOutput, true, kTileM>;
  typename Gemm::OutputCallbacks::Arguments callbacks{
      {
          {
              {
                  {
                      {
                          {},
                          {activation_scales.data_ptr<float>(), 0.0f, {}},
                          {},
                      },
                      {weight_scales.data_ptr<float>(), 0.0f, {}},
                      {},
                  },
                  {
                      bias.has_value() ? reinterpret_cast<ElementOutput const*>(bias->data_ptr()) : nullptr,
                      ElementOutput(0),
                      {},
                  },
                  {},
              },
              {
                  reinterpret_cast<ElementOutput const*>(gate->data_ptr()),
                  ElementOutput(1),
                  gate_span,
                  gate_row_stride,
              },
              {},
          },
          {
              reinterpret_cast<ElementOutput*>(residual->data_ptr()),
              ElementOutput(0),
              {int64_t(n), cute::_1{}, cute::_0{}},
          },
          {},
      },
      {
          reinterpret_cast<ElementOutput*>(output.data_ptr()),
          {int64_t(n), cute::_1{}, cute::_0{}},
      },
  };
  launch_fused_w4a4_gemm_impl<Gemm>(packed_activation, packed_weight, callbacks, stream);
}

template <typename ElementOutput>
void launch_fused_w4a4_gemm(
    torch::Tensor packed_activation, torch::Tensor packed_weight,
    torch::Tensor activation_scales, torch::Tensor weight_scales,
    std::optional<torch::Tensor> const& bias, std::optional<torch::Tensor> const& residual,
    std::optional<torch::Tensor> const& gate, int64_t gate_span,
    int64_t gate_row_stride, torch::Tensor output, cudaStream_t stream) {
  char const* tile = std::getenv("FASTWAM_W4A4_TILE");
  TORCH_CHECK(!tile || std::strcmp(tile, "auto") == 0 || std::strcmp(tile, "64") == 0 ||
              std::strcmp(tile, "128") == 0, "FASTWAM_W4A4_TILE must be auto, 64 or 128");
  bool small = tile && std::strcmp(tile, "64") == 0;
  if (!tile || std::strcmp(tile, "auto") == 0) {
    int64_t m = packed_activation.size(0), k = packed_activation.size(1) * 2;
    int64_t n = packed_weight.size(0);
    auto const* properties = at::cuda::getCurrentDeviceProperties();
    // Validated as one full-model configuration: 221.65 vs 241.74 ms for
    // ten official RHT DiT calls. Other shapes/architectures retain 128.
    bool measured =
        (m == 32 && ((k == 1024 && (n == 3072 || n == 4096)) ||
                     ((k == 3072 || k == 4096) && n == 1024))) ||
        (m == 129 && (k == 1024 || k == 3072) && n == 3072) ||
        (m == 294 && ((k == 3072 && (n == 3072 || n == 14336)) ||
                      (k == 14336 && n == 3072)));
    small = properties->major == 8 && properties->minor == 9 && measured;
  }
  if (small)
    launch_fused_w4a4_gemm_tiled<ElementOutput, 64>(packed_activation, packed_weight,
        activation_scales, weight_scales, bias, residual, gate, gate_span, gate_row_stride, output, stream);
  else
    launch_fused_w4a4_gemm_tiled<ElementOutput, 128>(packed_activation, packed_weight,
        activation_scales, weight_scales, bias, residual, gate, gate_span, gate_row_stride, output, stream);
}

template <typename scalar_t>
torch::Tensor run_dynamic(
    torch::Tensor x,
    torch::Tensor packed_weight,
    torch::Tensor weight_scales,
    std::optional<torch::Tensor> bias,
    std::optional<torch::Tensor> input_scale,
    bool input_scale_is_inverse,
    double clip_ratio,
    cudaStream_t stream,
    std::optional<torch::Tensor> rotation_signs = std::nullopt,
    std::optional<torch::Tensor> residual = std::nullopt,
    std::optional<torch::Tensor> gate = std::nullopt,
    int64_t gate_span = 1,
    int64_t gate_row_stride = 0,
    bool official_rotation = false) {
  int64_t input_features = x.size(-1);
  int64_t output_features = packed_weight.size(0);
  int64_t rows = x.numel() / input_features;
  auto packed_activation = torch::empty({rows, input_features / 2}, x.options().dtype(torch::kUInt8));
  auto activation_scales = torch::empty({rows}, x.options().dtype(torch::kFloat32));
  auto output = torch::empty({rows, output_features}, x.options());
  int quant_threads = input_features >= 4096 ? 512 : 256;

  if (rotation_signs.has_value()) {
    TORCH_CHECK(!input_scale.has_value(), "RHT W4A4 does not accept an additional input scale");
    if (official_rotation && input_features == 3072) {
      launch_official_rht_quantize<scalar_t, 12, 256>(
          x.data_ptr<scalar_t>(), rotation_signs->data_ptr<float>(), packed_activation.data_ptr<uint8_t>(),
          activation_scales.data_ptr<float>(), rows, static_cast<float>(clip_ratio), stream);
    } else if (official_rotation && input_features == 14336) {
      launch_official_rht_quantize<scalar_t, 28, 512>(
          x.data_ptr<scalar_t>(), rotation_signs->data_ptr<float>(), packed_activation.data_ptr<uint8_t>(),
          activation_scales.data_ptr<float>(), rows, static_cast<float>(clip_ratio), stream);
    } else dispatch_fused_rht_quantize(
        x.data_ptr<scalar_t>(), rotation_signs->data_ptr<float>(), packed_activation.data_ptr<uint8_t>(),
        activation_scales.data_ptr<float>(), rows, input_features, static_cast<float>(clip_ratio), stream);
  } else if (input_scale_is_inverse) {
    quantize_per_token_s4_kernel<scalar_t, true><<<rows, quant_threads, 0, stream>>>(
        x.data_ptr<scalar_t>(), input_scale.has_value() ? input_scale->data_ptr<float>() : nullptr,
        packed_activation.data_ptr<uint8_t>(), activation_scales.data_ptr<float>(),
        rows, input_features, static_cast<float>(clip_ratio));
  } else {
    quantize_per_token_s4_kernel<scalar_t, false><<<rows, quant_threads, 0, stream>>>(
        x.data_ptr<scalar_t>(), input_scale.has_value() ? input_scale->data_ptr<float>() : nullptr,
        packed_activation.data_ptr<uint8_t>(), activation_scales.data_ptr<float>(),
        rows, input_features, static_cast<float>(clip_ratio));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  using ElementOutput = std::conditional_t<
      std::is_same_v<scalar_t, at::Half>, cutlass::half_t, cutlass::bfloat16_t>;
  launch_fused_w4a4_gemm<ElementOutput>(
      packed_activation, packed_weight, activation_scales, weight_scales, bias,
      residual, gate, gate_span, gate_row_stride, output, stream);
  auto shape = x.sizes().vec();
  shape.back() = output_features;
  return output.view(shape);
}

template <typename scalar_t>
torch::Tensor run_dynamic_adaln(
    torch::Tensor x,
    torch::Tensor adaln_scale,
    torch::Tensor adaln_shift,
    int64_t modulation_span,
    double epsilon,
    torch::Tensor packed_weight,
    torch::Tensor weight_scales,
    std::optional<torch::Tensor> bias,
    std::optional<torch::Tensor> input_scale,
    bool input_scale_is_inverse,
    cudaStream_t stream) {
  int64_t k = x.size(-1);
  int64_t rows = x.numel() / k;
  int64_t n = packed_weight.size(0);
  auto packed_activation = torch::empty({rows, k / 2}, x.options().dtype(torch::kUInt8));
  auto activation_scales = torch::empty({rows}, x.options().dtype(torch::kFloat32));
  auto output = torch::empty({rows, n}, x.options());
  int quant_threads = k >= 4096 ? 512 : 256;
  int64_t modulation_row_stride = adaln_scale.stride(-2);
  size_t shared_bytes = static_cast<size_t>(k) * sizeof(float);
  if (input_scale_is_inverse) {
    quantize_dynamic_adaln_s4_kernel<scalar_t, true><<<rows, quant_threads, shared_bytes, stream>>>(
        x.data_ptr<scalar_t>(), adaln_scale.data_ptr<scalar_t>(), adaln_shift.data_ptr<scalar_t>(),
        modulation_row_stride, modulation_span, static_cast<float>(epsilon),
        input_scale.has_value() ? input_scale->data_ptr<float>() : nullptr,
        packed_activation.data_ptr<uint8_t>(), activation_scales.data_ptr<float>(), rows, k);
  } else {
    quantize_dynamic_adaln_s4_kernel<scalar_t, false><<<rows, quant_threads, shared_bytes, stream>>>(
        x.data_ptr<scalar_t>(), adaln_scale.data_ptr<scalar_t>(), adaln_shift.data_ptr<scalar_t>(),
        modulation_row_stride, modulation_span, static_cast<float>(epsilon),
        input_scale.has_value() ? input_scale->data_ptr<float>() : nullptr,
        packed_activation.data_ptr<uint8_t>(), activation_scales.data_ptr<float>(), rows, k);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  using ElementOutput = std::conditional_t<
      std::is_same_v<scalar_t, at::Half>, cutlass::half_t, cutlass::bfloat16_t>;
  launch_fused_w4a4_gemm<ElementOutput>(
      packed_activation, packed_weight, activation_scales, weight_scales, bias,
      std::nullopt, std::nullopt, 1, 0, output, stream);
  auto shape = x.sizes().vec();
  shape.back() = n;
  return output.view(shape);
}

template <typename scalar_t>
torch::Tensor run_static(
    torch::Tensor x,
    torch::Tensor packed_weight,
    torch::Tensor weight_scales,
    std::optional<torch::Tensor> bias,
    std::optional<torch::Tensor> input_scale,
    torch::Tensor activation_scale,
    torch::Tensor row_gains,
    bool input_scale_is_inverse,
    cudaStream_t stream) {
  int64_t k = x.size(-1);
  int64_t rows = x.numel() / k;
  int64_t n = packed_weight.size(0);
  auto packed_activation = torch::empty({rows, k / 2}, x.options().dtype(torch::kUInt8));
  auto row_scales = torch::empty({rows}, x.options().dtype(torch::kFloat32));
  auto output = torch::empty({rows, n}, x.options());
  int quant_threads = k >= 4096 ? 512 : 256;
  if (input_scale_is_inverse) {
    quantize_static_s4_kernel<scalar_t, true><<<rows, quant_threads, 0, stream>>>(
        x.data_ptr<scalar_t>(), input_scale.has_value() ? input_scale->data_ptr<float>() : nullptr,
        row_gains.data_ptr<scalar_t>(), activation_scale.data_ptr<float>(),
        packed_activation.data_ptr<uint8_t>(), row_scales.data_ptr<float>(), rows, k);
  } else {
    quantize_static_s4_kernel<scalar_t, false><<<rows, quant_threads, 0, stream>>>(
        x.data_ptr<scalar_t>(), input_scale.has_value() ? input_scale->data_ptr<float>() : nullptr,
        row_gains.data_ptr<scalar_t>(), activation_scale.data_ptr<float>(),
        packed_activation.data_ptr<uint8_t>(), row_scales.data_ptr<float>(), rows, k);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  using ElementOutput = std::conditional_t<
      std::is_same_v<scalar_t, at::Half>, cutlass::half_t, cutlass::bfloat16_t>;
  launch_fused_w4a4_gemm<ElementOutput>(
      packed_activation, packed_weight, row_scales, weight_scales, bias,
      std::nullopt, std::nullopt, 1, 0, output, stream);
  auto shape = x.sizes().vec();
  shape.back() = n;
  return output.view(shape);
}

template <typename scalar_t>
torch::Tensor run_static_streams(
    torch::Tensor x,
    torch::Tensor packed_weight,
    torch::Tensor weight_scales,
    std::optional<torch::Tensor> bias,
    std::optional<torch::Tensor> input_scale,
    torch::Tensor activation_scales,
    torch::Tensor stream_gains,
    int64_t stream_span,
    int64_t num_streams,
    int64_t schedule_index,
    bool input_scale_is_inverse,
    cudaStream_t stream,
    std::optional<torch::Tensor> residual = std::nullopt,
    std::optional<torch::Tensor> gate = std::nullopt,
    int64_t gate_span = 1,
    int64_t gate_row_stride = 0,
    std::optional<torch::Tensor> rotation_signs = std::nullopt) {
  int64_t k = x.size(-1);
  int64_t rows = x.numel() / k;
  int64_t n = packed_weight.size(0);
  auto packed_activation = torch::empty({rows, k / 2}, x.options().dtype(torch::kUInt8));
  auto row_scales = torch::empty({rows}, x.options().dtype(torch::kFloat32));
  auto output = torch::empty({rows, n}, x.options());
  float const* selected_scale = activation_scales.data_ptr<float>() + schedule_index;
  float const* selected_gains = stream_gains.data_ptr<float>() + schedule_index * num_streams;
  if (rotation_signs.has_value()) {
    TORCH_CHECK(input_scale.has_value() && input_scale_is_inverse,
                "fused RHT WAM requires a prepared inverse input scale");
    dispatch_official_wam_quantize(
        x.data_ptr<scalar_t>(), rotation_signs->data_ptr<float>(), input_scale->data_ptr<float>(),
        selected_gains, selected_scale,
        packed_activation.data_ptr<uint8_t>(), row_scales.data_ptr<float>(),
        rows, k, stream_span, num_streams, x.size(-2), stream);
  } else if (input_scale_is_inverse) {
    int quant_threads = k >= 4096 ? 512 : 256;
    quantize_static_stream_s4_kernel<scalar_t, true><<<rows, quant_threads, 0, stream>>>(
        x.data_ptr<scalar_t>(), input_scale.has_value() ? input_scale->data_ptr<float>() : nullptr,
        selected_gains, selected_scale, packed_activation.data_ptr<uint8_t>(), row_scales.data_ptr<float>(),
        rows, k, stream_span, num_streams, x.size(-2));
  } else {
    int quant_threads = k >= 4096 ? 512 : 256;
    quantize_static_stream_s4_kernel<scalar_t, false><<<rows, quant_threads, 0, stream>>>(
        x.data_ptr<scalar_t>(), input_scale.has_value() ? input_scale->data_ptr<float>() : nullptr,
        selected_gains, selected_scale, packed_activation.data_ptr<uint8_t>(), row_scales.data_ptr<float>(),
        rows, k, stream_span, num_streams, x.size(-2));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  using ElementOutput = std::conditional_t<
      std::is_same_v<scalar_t, at::Half>, cutlass::half_t, cutlass::bfloat16_t>;
  launch_fused_w4a4_gemm<ElementOutput>(
      packed_activation, packed_weight, row_scales, weight_scales, bias,
      residual, gate, gate_span, gate_row_stride, output, stream);
  auto shape = x.sizes().vec();
  shape.back() = n;
  return output.view(shape);
}

template <typename scalar_t>
torch::Tensor run_static_streams_adaln(
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
    int64_t num_streams,
    int64_t schedule_index,
    bool input_scale_is_inverse,
    cudaStream_t stream,
    std::optional<torch::Tensor> rotation_signs = std::nullopt) {
  int64_t k = x.size(-1);
  int64_t rows = x.numel() / k;
  int64_t n = packed_weight.size(0);
  auto packed_activation = torch::empty({rows, k / 2}, x.options().dtype(torch::kUInt8));
  auto row_scales = torch::empty({rows}, x.options().dtype(torch::kFloat32));
  auto output = torch::empty({rows, n}, x.options());
  TORCH_CHECK(stream_span > 0, "WAM AdaLN producer requires equal-length streams");
  int64_t modulation_row_stride = adaln_scale.stride(-2);
  float const* selected_scale = activation_scales.data_ptr<float>() + schedule_index;
  float const* selected_gains = stream_gains.data_ptr<float>() + schedule_index * num_streams;
  if (rotation_signs.has_value()) {
    TORCH_CHECK(input_scale.has_value() && input_scale_is_inverse,
                "fused RHT WAM AdaLN requires a prepared inverse input scale");
    dispatch_fused_rht_wam_quantize<true>(
        x.data_ptr<scalar_t>(), rotation_signs->data_ptr<float>(), input_scale->data_ptr<float>(),
        adaln_scale.data_ptr<scalar_t>(), adaln_shift.data_ptr<scalar_t>(),
        modulation_row_stride, modulation_span, static_cast<float>(epsilon), selected_gains, selected_scale,
        packed_activation.data_ptr<uint8_t>(), row_scales.data_ptr<float>(),
        rows, k, stream_span, num_streams, stream);
  } else if (input_scale_is_inverse) {
    int quant_threads = k >= 4096 ? 512 : 256;
    quantize_static_adaln_stream_s4_kernel<scalar_t, true><<<rows, quant_threads, 0, stream>>>(
        x.data_ptr<scalar_t>(), adaln_scale.data_ptr<scalar_t>(), adaln_shift.data_ptr<scalar_t>(),
        modulation_row_stride, modulation_span, static_cast<float>(epsilon),
        input_scale.has_value() ? input_scale->data_ptr<float>() : nullptr,
        selected_gains, selected_scale, packed_activation.data_ptr<uint8_t>(), row_scales.data_ptr<float>(),
        rows, k, stream_span, num_streams);
  } else {
    int quant_threads = k >= 4096 ? 512 : 256;
    quantize_static_adaln_stream_s4_kernel<scalar_t, false><<<rows, quant_threads, 0, stream>>>(
        x.data_ptr<scalar_t>(), adaln_scale.data_ptr<scalar_t>(), adaln_shift.data_ptr<scalar_t>(),
        modulation_row_stride, modulation_span, static_cast<float>(epsilon),
        input_scale.has_value() ? input_scale->data_ptr<float>() : nullptr,
        selected_gains, selected_scale, packed_activation.data_ptr<uint8_t>(), row_scales.data_ptr<float>(),
        rows, k, stream_span, num_streams);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  using ElementOutput = std::conditional_t<
      std::is_same_v<scalar_t, at::Half>, cutlass::half_t, cutlass::bfloat16_t>;
  launch_fused_w4a4_gemm<ElementOutput>(
      packed_activation, packed_weight, row_scales, weight_scales, bias,
      std::nullopt, std::nullopt, 1, 0, output, stream);
  auto shape = x.sizes().vec();
  shape.back() = n;
  return output.view(shape);
}

void check_inputs(
    torch::Tensor const& x,
    torch::Tensor const& packed_weight,
    torch::Tensor const& weight_scales,
    std::optional<torch::Tensor> const& bias,
    std::optional<torch::Tensor> const& input_scale) {
  TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.dim() >= 2, "x must be contiguous CUDA with rank >= 2");
  TORCH_CHECK(x.scalar_type() == torch::kFloat16 || x.scalar_type() == torch::kBFloat16, "x must be FP16/BF16");
  TORCH_CHECK(packed_weight.is_cuda() && packed_weight.is_contiguous() &&
                  packed_weight.scalar_type() == torch::kUInt8 && packed_weight.dim() == 2,
              "weight must be contiguous packed uint8 CUDA");
  TORCH_CHECK(weight_scales.is_cuda() && weight_scales.is_contiguous() &&
                  weight_scales.scalar_type() == torch::kFloat32,
              "weight scales must be contiguous float32 CUDA");
  int64_t k = x.size(-1);
  int64_t n = packed_weight.size(0);
  TORCH_CHECK(k % 256 == 0, "native W4A4 requires K divisible by 256");
  TORCH_CHECK(packed_weight.size(1) * 2 == k, "packed weight K mismatch");
  TORCH_CHECK(n % 8 == 0, "native W4A4 requires N divisible by 8");
  TORCH_CHECK(weight_scales.numel() == n, "weight scale count mismatch");
  TORCH_CHECK(x.device() == packed_weight.device() && x.device() == weight_scales.device(),
              "all W4A4 operands must be on the same CUDA device");
  if (bias.has_value()) {
    TORCH_CHECK(bias->is_cuda() && bias->is_contiguous() && bias->device() == x.device() &&
                    bias->scalar_type() == x.scalar_type() && bias->numel() == n,
                "bias must be contiguous CUDA with the input dtype and N values");
  }
  if (input_scale.has_value()) {
    TORCH_CHECK(input_scale->is_cuda() && input_scale->is_contiguous() && input_scale->device() == x.device() &&
                    input_scale->scalar_type() == torch::kFloat32 && input_scale->numel() == k,
                "input scale must be contiguous float32 CUDA with K values");
  }
}

void check_adaln_inputs(
    torch::Tensor const& x,
    torch::Tensor const& adaln_scale,
    torch::Tensor const& adaln_shift,
    int64_t modulation_span,
    double epsilon) {
  int64_t k = x.size(-1);
  int64_t rows = x.numel() / k;
  TORCH_CHECK(epsilon > 0.0, "LayerNorm epsilon must be positive");
  TORCH_CHECK(adaln_scale.is_cuda() && adaln_shift.is_cuda() &&
                  adaln_scale.device() == x.device() && adaln_shift.device() == x.device(),
              "AdaLN tensors must be on the input CUDA device");
  TORCH_CHECK(adaln_scale.scalar_type() == x.scalar_type() && adaln_shift.scalar_type() == x.scalar_type(),
              "AdaLN tensors must match the input dtype");
  TORCH_CHECK(adaln_scale.dim() >= 2 && adaln_scale.size(-1) == k &&
                  adaln_shift.sizes() == adaln_scale.sizes(),
              "AdaLN scale and shift must have matching [..., K] shapes");
  TORCH_CHECK(adaln_scale.stride(-1) == 1 && adaln_shift.stride(-1) == 1 &&
                  adaln_scale.stride(-2) == adaln_shift.stride(-2),
              "AdaLN tensors need unit inner stride and matching row strides");
  TORCH_CHECK(modulation_span > 0 && rows == (adaln_scale.numel() / k) * modulation_span,
              "AdaLN rows and modulation_span do not cover the input rows");
}

void check_residual_gate_inputs(
    torch::Tensor const& x,
    torch::Tensor const& packed_weight,
    torch::Tensor const& residual,
    torch::Tensor const& gate,
    int64_t gate_span) {
  int64_t rows = x.numel() / x.size(-1);
  int64_t n = packed_weight.size(0);
  TORCH_CHECK(residual.is_cuda() && residual.is_contiguous() && residual.device() == x.device() &&
                  residual.scalar_type() == x.scalar_type() && residual.numel() == rows * n,
              "residual must be contiguous CUDA with one input-dtype value per output element");
  TORCH_CHECK(gate.is_cuda() && gate.device() == x.device() && gate.scalar_type() == x.scalar_type() &&
                  gate.dim() >= 2 && gate.size(-1) == n && gate.stride(-1) == 1,
              "gate must be CUDA [..., N] with the input dtype and unit inner stride");
  TORCH_CHECK(gate_span > 0 && rows == (gate.numel() / n) * gate_span,
              "gate rows and gate_span do not cover the output rows");
}

void check_static_schedule(
    torch::Tensor const& x,
    torch::Tensor const& activation_scales,
    torch::Tensor const& stream_gains,
    int64_t stream_span,
    int64_t schedule_index) {
  int64_t rows = x.numel() / x.size(-1);
  TORCH_CHECK(activation_scales.is_cuda() && activation_scales.is_contiguous() &&
                  activation_scales.device() == x.device() && activation_scales.scalar_type() == torch::kFloat32 &&
                  activation_scales.dim() == 1 && activation_scales.numel() > 0,
              "activation_scales must be a non-empty contiguous CUDA float32 vector");
  TORCH_CHECK(stream_gains.is_cuda() && stream_gains.is_contiguous() && stream_gains.device() == x.device() &&
                  stream_gains.scalar_type() == torch::kFloat32 && stream_gains.dim() == 2 &&
                  stream_gains.size(0) == activation_scales.numel() && stream_gains.size(1) > 0,
              "stream_gains must be contiguous CUDA float32 [timesteps, streams]");
  TORCH_CHECK(schedule_index >= 0 && schedule_index < activation_scales.numel(),
              "WAM schedule index is out of range");
  TORCH_CHECK((stream_span > 0 && x.size(-2) == stream_span * stream_gains.size(1)) ||
                  (stream_span < 0 && stream_gains.size(1) == 2 && -stream_span < x.size(-2)),
              "expected equal semantic streams or a negative proprio suffix span");
}

void check_rotation_signs(
    torch::Tensor const& x,
    torch::Tensor const& rotation_signs,
    std::optional<torch::Tensor> const& input_scale,
    bool input_scale_is_inverse) {
  TORCH_CHECK(rotation_signs.is_cuda() && rotation_signs.is_contiguous() &&
                  rotation_signs.device() == x.device() &&
                  rotation_signs.scalar_type() == torch::kFloat32 &&
                  rotation_signs.numel() == x.size(-1),
              "rotation_signs must be contiguous CUDA float32 with K values");
  TORCH_CHECK(input_scale.has_value() && input_scale_is_inverse,
              "fused RHT WAM requires a prepared inverse input scale");
}

}  // namespace

torch::Tensor w4a4_dynamic_per_token_cuda(
    torch::Tensor x,
    torch::Tensor packed_weight,
    torch::Tensor weight_scales,
    std::optional<torch::Tensor> bias,
    double clip_ratio) {
  TORCH_CHECK(x.is_cuda() && x.is_contiguous(), "x must be contiguous CUDA");
  TORCH_CHECK(x.scalar_type() == torch::kFloat16 || x.scalar_type() == torch::kBFloat16, "x must be FP16/BF16");
  TORCH_CHECK(packed_weight.is_cuda() && packed_weight.scalar_type() == torch::kUInt8, "weight must be packed uint8 CUDA");
  TORCH_CHECK(weight_scales.is_cuda() && weight_scales.scalar_type() == torch::kFloat32, "scales must be float32 CUDA");
  TORCH_CHECK(x.size(-1) % 256 == 0, "K must be divisible by 256");
  TORCH_CHECK(packed_weight.size(1) * 2 == x.size(-1), "packed weight K mismatch");
  TORCH_CHECK(packed_weight.size(0) % 8 == 0, "N must be divisible by 8");
  TORCH_CHECK(weight_scales.numel() == packed_weight.size(0), "weight scale count mismatch");
  TORCH_CHECK(clip_ratio > 0.0 && clip_ratio <= 1.0, "clip ratio must be in (0, 1]");
  if (bias.has_value()) {
    TORCH_CHECK(bias->is_cuda() && bias->is_contiguous() && bias->scalar_type() == x.scalar_type(), "bias mismatch");
    TORCH_CHECK(bias->numel() == packed_weight.size(0), "bias size mismatch");
  }
  c10::cuda::CUDAGuard guard(x.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (x.scalar_type() == torch::kFloat16) {
    return run_dynamic<at::Half>(
        x, packed_weight, weight_scales, bias, std::nullopt, false, clip_ratio, stream);
  }
  return run_dynamic<at::BFloat16>(
      x, packed_weight, weight_scales, bias, std::nullopt, false, clip_ratio, stream);
}

torch::Tensor w4a4_rht_dynamic_per_token_cuda(
    torch::Tensor x,
    torch::Tensor rotation_signs,
    torch::Tensor packed_weight,
    torch::Tensor weight_scales,
    std::optional<torch::Tensor> bias,
    double clip_ratio) {
  TORCH_CHECK(x.is_cuda() && x.is_contiguous(), "x must be contiguous CUDA");
  TORCH_CHECK(x.scalar_type() == torch::kFloat16 || x.scalar_type() == torch::kBFloat16, "x must be FP16/BF16");
  TORCH_CHECK(rotation_signs.is_cuda() && rotation_signs.is_contiguous(), "rotation signs must be contiguous CUDA");
  TORCH_CHECK(rotation_signs.scalar_type() == torch::kFloat32, "rotation signs must be float32");
  TORCH_CHECK(rotation_signs.numel() == x.size(-1), "rotation sign count mismatch");
  TORCH_CHECK(packed_weight.is_cuda() && packed_weight.scalar_type() == torch::kUInt8, "weight must be packed uint8 CUDA");
  TORCH_CHECK(weight_scales.is_cuda() && weight_scales.scalar_type() == torch::kFloat32, "scales must be float32 CUDA");
  TORCH_CHECK(packed_weight.size(1) * 2 == x.size(-1), "packed weight K mismatch");
  TORCH_CHECK(packed_weight.size(0) % 8 == 0, "N must be divisible by 8");
  TORCH_CHECK(weight_scales.numel() == packed_weight.size(0), "weight scale count mismatch");
  TORCH_CHECK(clip_ratio > 0.0 && clip_ratio <= 1.0, "clip ratio must be in (0, 1]");
  if (bias.has_value()) {
    TORCH_CHECK(bias->is_cuda() && bias->is_contiguous() && bias->scalar_type() == x.scalar_type(), "bias mismatch");
    TORCH_CHECK(bias->numel() == packed_weight.size(0), "bias size mismatch");
  }
  c10::cuda::CUDAGuard guard(x.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (x.scalar_type() == torch::kFloat16) {
    return run_dynamic<at::Half>(
        x, packed_weight, weight_scales, bias, std::nullopt, false, clip_ratio, stream, rotation_signs);
  }
  return run_dynamic<at::BFloat16>(
      x, packed_weight, weight_scales, bias, std::nullopt, false, clip_ratio, stream, rotation_signs);
}

torch::Tensor w4a4_rht_official_dynamic_per_token_cuda(
    torch::Tensor x, torch::Tensor rotation_signs, torch::Tensor packed_weight,
    torch::Tensor weight_scales, std::optional<torch::Tensor> bias, double clip_ratio) {
  check_inputs(x, packed_weight, weight_scales, bias, std::nullopt);
  TORCH_CHECK(rotation_signs.is_cuda() && rotation_signs.is_contiguous() &&
              rotation_signs.device() == x.device() && rotation_signs.scalar_type() == torch::kFloat32 &&
              rotation_signs.numel() == x.size(-1), "rotation signs must be contiguous CUDA float32 [K]");
  int64_t k = x.size(-1);
  TORCH_CHECK(((k & (k - 1)) == 0 && k >= 256 && k <= 8192) || k == 3072 || k == 14336,
              "Unsupported official RHT width");
  TORCH_CHECK(clip_ratio > 0.0 && clip_ratio <= 1.0, "clip ratio must be in (0, 1]");
  c10::cuda::CUDAGuard guard(x.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (x.scalar_type() == torch::kFloat16)
    return run_dynamic<at::Half>(x, packed_weight, weight_scales, bias, std::nullopt, false,
        clip_ratio, stream, rotation_signs, std::nullopt, std::nullopt, 1, 0, true);
  return run_dynamic<at::BFloat16>(x, packed_weight, weight_scales, bias, std::nullopt, false,
      clip_ratio, stream, rotation_signs, std::nullopt, std::nullopt, 1, 0, true);
}

torch::Tensor w4a4_symmetric_dynamic_cuda(
    torch::Tensor x,
    torch::Tensor packed_weight,
    torch::Tensor weight_scales,
    std::optional<torch::Tensor> bias,
    std::optional<torch::Tensor> input_scale,
    bool input_scale_is_inverse) {
  check_inputs(x, packed_weight, weight_scales, bias, input_scale);
  c10::cuda::CUDAGuard guard(x.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (x.scalar_type() == torch::kFloat16) {
    return run_dynamic<at::Half>(
        x, packed_weight, weight_scales, bias, input_scale, input_scale_is_inverse, 1.0, stream);
  }
  return run_dynamic<at::BFloat16>(
      x, packed_weight, weight_scales, bias, input_scale, input_scale_is_inverse, 1.0, stream);
}

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
    bool input_scale_is_inverse) {
  check_inputs(x, packed_weight, weight_scales, bias, input_scale);
  check_adaln_inputs(x, adaln_scale, adaln_shift, modulation_span, epsilon);
  c10::cuda::CUDAGuard guard(x.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (x.scalar_type() == torch::kFloat16) {
    return run_dynamic_adaln<at::Half>(
        x, adaln_scale, adaln_shift, modulation_span, epsilon,
        packed_weight, weight_scales, bias, input_scale, input_scale_is_inverse, stream);
  }
  return run_dynamic_adaln<at::BFloat16>(
      x, adaln_scale, adaln_shift, modulation_span, epsilon,
      packed_weight, weight_scales, bias, input_scale, input_scale_is_inverse, stream);
}

torch::Tensor w4a4_symmetric_dynamic_gate_residual_cuda(
    torch::Tensor x,
    torch::Tensor packed_weight,
    torch::Tensor weight_scales,
    std::optional<torch::Tensor> bias,
    std::optional<torch::Tensor> input_scale,
    bool input_scale_is_inverse,
    torch::Tensor residual,
    torch::Tensor gate,
    int64_t gate_span) {
  check_inputs(x, packed_weight, weight_scales, bias, input_scale);
  check_residual_gate_inputs(x, packed_weight, residual, gate, gate_span);
  c10::cuda::CUDAGuard guard(x.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  int64_t gate_row_stride = gate.stride(-2);
  if (x.scalar_type() == torch::kFloat16) {
    return run_dynamic<at::Half>(
        x, packed_weight, weight_scales, bias, input_scale, input_scale_is_inverse, 1.0, stream,
        std::nullopt, residual, gate, gate_span, gate_row_stride);
  }
  return run_dynamic<at::BFloat16>(
      x, packed_weight, weight_scales, bias, input_scale, input_scale_is_inverse, 1.0, stream,
      std::nullopt, residual, gate, gate_span, gate_row_stride);
}

torch::Tensor w4a4_symmetric_static_cuda(
    torch::Tensor x,
    torch::Tensor packed_weight,
    torch::Tensor weight_scales,
    std::optional<torch::Tensor> bias,
    std::optional<torch::Tensor> input_scale,
    torch::Tensor activation_scale,
    torch::Tensor row_gains,
    bool input_scale_is_inverse) {
  check_inputs(x, packed_weight, weight_scales, bias, input_scale);
  int64_t rows = x.numel() / x.size(-1);
  TORCH_CHECK(activation_scale.is_cuda() && activation_scale.device() == x.device() &&
                  activation_scale.scalar_type() == torch::kFloat32 && activation_scale.numel() == 1,
              "activation_scale must be one CUDA float32 value");
  TORCH_CHECK(row_gains.is_cuda() && row_gains.is_contiguous() && row_gains.device() == x.device() &&
                  row_gains.scalar_type() == x.scalar_type() && row_gains.numel() == rows,
              "row_gains must be contiguous CUDA with one input-dtype value per flattened row");
  c10::cuda::CUDAGuard guard(x.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (x.scalar_type() == torch::kFloat16) {
    return run_static<at::Half>(
        x, packed_weight, weight_scales, bias, input_scale,
        activation_scale, row_gains, input_scale_is_inverse, stream);
  }
  return run_static<at::BFloat16>(
      x, packed_weight, weight_scales, bias, input_scale,
      activation_scale, row_gains, input_scale_is_inverse, stream);
}

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
    bool input_scale_is_inverse) {
  check_inputs(x, packed_weight, weight_scales, bias, input_scale);
  check_static_schedule(x, activation_scales, stream_gains, stream_span, schedule_index);
  c10::cuda::CUDAGuard guard(x.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  int64_t num_streams = stream_gains.size(1);
  if (x.scalar_type() == torch::kFloat16) {
    return run_static_streams<at::Half>(
        x, packed_weight, weight_scales, bias, input_scale, activation_scales, stream_gains,
        stream_span, num_streams, schedule_index, input_scale_is_inverse, stream);
  }
  return run_static_streams<at::BFloat16>(
      x, packed_weight, weight_scales, bias, input_scale, activation_scales, stream_gains,
      stream_span, num_streams, schedule_index, input_scale_is_inverse, stream);
}

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
    bool input_scale_is_inverse) {
  check_inputs(x, packed_weight, weight_scales, bias, input_scale);
  check_static_schedule(x, activation_scales, stream_gains, stream_span, schedule_index);
  check_rotation_signs(x, rotation_signs, input_scale, input_scale_is_inverse);
  c10::cuda::CUDAGuard guard(x.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  int64_t num_streams = stream_gains.size(1);
  if (x.scalar_type() == torch::kFloat16) {
    return run_static_streams<at::Half>(
        x, packed_weight, weight_scales, bias, input_scale, activation_scales, stream_gains,
        stream_span, num_streams, schedule_index, input_scale_is_inverse, stream,
        std::nullopt, std::nullopt, 1, 0, rotation_signs);
  }
  return run_static_streams<at::BFloat16>(
      x, packed_weight, weight_scales, bias, input_scale, activation_scales, stream_gains,
      stream_span, num_streams, schedule_index, input_scale_is_inverse, stream,
      std::nullopt, std::nullopt, 1, 0, rotation_signs);
}

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
    bool input_scale_is_inverse) {
  check_inputs(x, packed_weight, weight_scales, bias, input_scale);
  check_adaln_inputs(x, adaln_scale, adaln_shift, modulation_span, epsilon);
  check_static_schedule(x, activation_scales, stream_gains, stream_span, schedule_index);
  c10::cuda::CUDAGuard guard(x.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  int64_t num_streams = stream_gains.size(1);
  if (x.scalar_type() == torch::kFloat16) {
    return run_static_streams_adaln<at::Half>(
        x, adaln_scale, adaln_shift, modulation_span, epsilon,
        packed_weight, weight_scales, bias, input_scale, activation_scales, stream_gains,
        stream_span, num_streams, schedule_index, input_scale_is_inverse, stream);
  }
  return run_static_streams_adaln<at::BFloat16>(
      x, adaln_scale, adaln_shift, modulation_span, epsilon,
      packed_weight, weight_scales, bias, input_scale, activation_scales, stream_gains,
      stream_span, num_streams, schedule_index, input_scale_is_inverse, stream);
}

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
    bool input_scale_is_inverse) {
  check_inputs(x, packed_weight, weight_scales, bias, input_scale);
  check_adaln_inputs(x, adaln_scale, adaln_shift, modulation_span, epsilon);
  check_static_schedule(x, activation_scales, stream_gains, stream_span, schedule_index);
  check_rotation_signs(x, rotation_signs, input_scale, input_scale_is_inverse);
  c10::cuda::CUDAGuard guard(x.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  int64_t num_streams = stream_gains.size(1);
  if (x.scalar_type() == torch::kFloat16) {
    return run_static_streams_adaln<at::Half>(
        x, adaln_scale, adaln_shift, modulation_span, epsilon,
        packed_weight, weight_scales, bias, input_scale, activation_scales, stream_gains,
        stream_span, num_streams, schedule_index, input_scale_is_inverse, stream, rotation_signs);
  }
  return run_static_streams_adaln<at::BFloat16>(
      x, adaln_scale, adaln_shift, modulation_span, epsilon,
      packed_weight, weight_scales, bias, input_scale, activation_scales, stream_gains,
      stream_span, num_streams, schedule_index, input_scale_is_inverse, stream, rotation_signs);
}

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
    int64_t gate_span) {
  check_inputs(x, packed_weight, weight_scales, bias, input_scale);
  check_static_schedule(x, activation_scales, stream_gains, stream_span, schedule_index);
  check_residual_gate_inputs(x, packed_weight, residual, gate, gate_span);
  c10::cuda::CUDAGuard guard(x.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  int64_t num_streams = stream_gains.size(1);
  int64_t gate_row_stride = gate.stride(-2);
  if (x.scalar_type() == torch::kFloat16) {
    return run_static_streams<at::Half>(
        x, packed_weight, weight_scales, bias, input_scale, activation_scales, stream_gains,
        stream_span, num_streams, schedule_index, input_scale_is_inverse, stream,
        residual, gate, gate_span, gate_row_stride);
  }
  return run_static_streams<at::BFloat16>(
      x, packed_weight, weight_scales, bias, input_scale, activation_scales, stream_gains,
      stream_span, num_streams, schedule_index, input_scale_is_inverse, stream,
      residual, gate, gate_span, gate_row_stride);
}

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
    int64_t gate_span) {
  check_inputs(x, packed_weight, weight_scales, bias, input_scale);
  check_static_schedule(x, activation_scales, stream_gains, stream_span, schedule_index);
  check_rotation_signs(x, rotation_signs, input_scale, input_scale_is_inverse);
  check_residual_gate_inputs(x, packed_weight, residual, gate, gate_span);
  c10::cuda::CUDAGuard guard(x.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  int64_t num_streams = stream_gains.size(1);
  int64_t gate_row_stride = gate.stride(-2);
  if (x.scalar_type() == torch::kFloat16) {
    return run_static_streams<at::Half>(
        x, packed_weight, weight_scales, bias, input_scale, activation_scales, stream_gains,
        stream_span, num_streams, schedule_index, input_scale_is_inverse, stream,
        residual, gate, gate_span, gate_row_stride, rotation_signs);
  }
  return run_static_streams<at::BFloat16>(
      x, packed_weight, weight_scales, bias, input_scale, activation_scales, stream_gains,
      stream_span, num_streams, schedule_index, input_scale_is_inverse, stream,
      residual, gate, gate_span, gate_row_stride, rotation_signs);
}
