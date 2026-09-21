#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <optional>
#include <string>
#include <type_traits>

#include "cutlass/cutlass.h"
#include "cutlass/device_kernel.h"
#include "cutlass/epilogue/threadblock/default_epilogue_tensor_op.h"
#include "cutlass/epilogue/threadblock/epilogue_with_visitor_callbacks.h"
#include "cutlass/epilogue/threadblock/fusion/visitors.hpp"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal_with_visitor.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm80.h"
#include "cutlass/gemm/threadblock/mma_pipelined.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/numeric_types.h"
#include "cutlass/transform/threadblock/predicated_tile_iterator.h"
#include "cutlass/transform/threadblock/regular_tile_iterator_tensor_op.h"

namespace {

using ElementA = int8_t;
using ElementB = int8_t;
using PackedElementB = cutlass::int4b_t;
using ElementAccumulator = int32_t;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;

// FastWAM inference has M=32/129/294. The SM89 path tunes both CTA and warp
// geometry: 64x128 CTAs with 32x64 warps outperform the Cosmos 256x128/64x64
// layout on the measured FastWAM shapes. The old 64x64/64x64 experiment is
// retained separately; reducing only that CTA's size had not improved latency.
template <int TileM, int TileN, int WarpM = 64, int TileK = 128>
struct W4A8Tile {
  using ThreadblockShape = cutlass::gemm::GemmShape<TileM, TileN, TileK>;
  using WarpShape = cutlass::gemm::GemmShape<WarpM, 64, TileK>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;
  using MmaCore = cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementAccumulator, LayoutC,
      cutlass::arch::OpClassTensorOp, 2, cutlass::arch::OpMultiplyAddSaturate>;
  using IteratorA = cutlass::transform::threadblock::PredicatedTileIterator<
      cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
      ElementA, LayoutA, 1, typename MmaCore::IteratorThreadMapA, 16>;
  // A packed INT4 fragment is expanded to INT8 before MMA shared-memory use.
  using IteratorB = cutlass::transform::threadblock::PredicatedTileIterator<
      cutlass::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>,
      PackedElementB, LayoutB, 0, typename MmaCore::IteratorThreadMapB, 16>;
  using SmemIteratorA = cutlass::transform::threadblock::RegularTileIterator<
      cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
      ElementA, typename MmaCore::SmemLayoutA, 0, typename MmaCore::IteratorThreadMapA>;
  using SmemIteratorB = cutlass::transform::threadblock::RegularTileIterator<
      cutlass::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>,
      ElementB, typename MmaCore::SmemLayoutB, 1, typename MmaCore::IteratorThreadMapB>;
  using ThreadblockMma = cutlass::gemm::threadblock::MmaPipelined<
      typename MmaCore::Shape, IteratorA, SmemIteratorA, IteratorB, SmemIteratorB,
      ElementAccumulator, LayoutC, typename MmaCore::MmaPolicy>;
  static int const kPartitionsK = ThreadblockShape::kK / WarpShape::kK;
};

using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

// Load a [groups, N] gate without expanding it to [M, N]. Consecutive
// ``group_span`` output rows share one gate row. ``gate_row_stride`` also
// permits AdaLN chunk views whose logical rows are separated in a wider [*, 3N]
// tensor. Keeping this lookup in the epilogue removes both the expanded gate
// tensor and the post-GEMM multiply/add kernel.
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
  static constexpr Params to_underlying_arguments(
      ProblemShape const&, Arguments const& args, void*) {
    return args;
  }

  template <class ProblemShape>
  static size_t get_workspace_size(ProblemShape const&, Arguments const&) {
    return 0;
  }

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
        for (int i = 0; i < cute::size(dst_elements); ++i) {
          dst_elements(i) = params_ptr->null_default;
        }
        return;
      }
      int64_t n = cute::get<1>(problem_shape);
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
    // This tensor is used only to obtain the standard epilogue partition
    // shape. The actual grouped address is computed from identity coordinates.
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

// CUTLASS's stock VisitorAuxLoad assumes a non-null pointer. This nullable
// variant supplies the additive identity for ordinary (non-residual) Linears
// while loading the full [M, N] residual for fused DiT output projections.
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
  static size_t get_workspace_size(ProblemShape const&, Arguments const&) {
    return 0;
  }

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
        for (int i = 0; i < cute::size(dst_elements); ++i) {
          dst_elements(i) = params_ptr->null_default;
        }
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
    auto mAux = cute::make_tensor(
        cute::make_gmem_ptr(params_ptr->ptr_aux), problem_shape, params_ptr->dAux);
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

// The visitor epilogue consumes INT32 accumulator fragments inside the GEMM
// kernel, before any global-memory accumulator store. It broadcasts one
// activation scale per row and one weight scale plus bias per output column,
// then stores FP16/BF16 directly. Consequently no [M, N] INT32 tensor is
// materialized in global memory.
template <typename ElementOutput, int TileM = 256, int TileN = 128, int WarpM = 64, int TileK = 128>
struct PlainFusedEpilogueGemm {
  using ThreadblockShape = typename W4A8Tile<TileM, TileN, WarpM, TileK>::ThreadblockShape;
  using WarpShape = typename W4A8Tile<TileM, TileN, WarpM, TileK>::WarpShape;
  using ThreadblockMma = typename W4A8Tile<TileM, TileN, WarpM, TileK>::ThreadblockMma;
  static int const kPartitionsK = W4A8Tile<TileM, TileN, WarpM, TileK>::kPartitionsK;
  static int const kElementsPerAccess = 8;
  static int const kEpilogueStages = 1;

  using OutputTileThreadMap = cutlass::epilogue::threadblock::OutputTileThreadLayout<
      ThreadblockShape,
      WarpShape,
      ElementOutput,
      kElementsPerAccess,
      kEpilogueStages>;
  using Accumulator = cutlass::epilogue::threadblock::VisitorAccFetch;
  using ActivationScale = cutlass::epilogue::threadblock::VisitorColBroadcast<
      OutputTileThreadMap,
      float>;
  using Multiply = cutlass::epilogue::threadblock::VisitorCompute<
      cutlass::multiplies,
      float,
      float,
      cutlass::FloatRoundStyle::round_to_nearest>;
  using ScaleActivation = cutlass::epilogue::threadblock::Sm80EVT<
      Multiply,
      Accumulator,
      ActivationScale>;
  using WeightScale = cutlass::epilogue::threadblock::VisitorRowBroadcast<
      OutputTileThreadMap,
      float,
      cute::Stride<cute::_0, cute::_1, cute::_0>>;
  using ScaleWeight = cutlass::epilogue::threadblock::Sm80EVT<
      Multiply,
      ScaleActivation,
      WeightScale>;
  using Bias = cutlass::epilogue::threadblock::VisitorRowBroadcast<
      OutputTileThreadMap,
      ElementOutput,
      cute::Stride<cute::_0, cute::_1, cute::_0>>;
  using Add = cutlass::epilogue::threadblock::VisitorCompute<
      cutlass::plus,
      float,
      float,
      cutlass::FloatRoundStyle::round_to_nearest>;
  using AddBias = cutlass::epilogue::threadblock::Sm80EVT<Add, ScaleWeight, Bias>;
  using Output = cutlass::epilogue::threadblock::VisitorAuxStore<
      OutputTileThreadMap,
      ElementOutput,
      cutlass::FloatRoundStyle::round_to_nearest,
      cute::Stride<int64_t, cute::_1, cute::_0>>;
  using OutputCallbacks = cutlass::epilogue::threadblock::Sm80EVT<Output, AddBias>;
  using ThreadMapOutputOp = cutlass::epilogue::thread::LinearCombination<
      ElementOutput,
      kElementsPerAccess,
      ElementAccumulator,
      float>;
  using DefaultEpilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueTensorOp<
      ThreadblockShape,
      typename ThreadblockMma::Operator,
      kPartitionsK,
      ThreadMapOutputOp,
      kElementsPerAccess>::Epilogue;
  using Epilogue = cutlass::epilogue::threadblock::EpilogueWithVisitorCallbacks<
      DefaultEpilogue,
      OutputCallbacks,
      kEpilogueStages>;
  using Kernel = cutlass::gemm::kernel::GemmWithEpilogueVisitor<
      ThreadblockMma,
      Epilogue,
      ThreadblockSwizzle>;
  using DeviceGemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
};

template <typename ElementOutput, int TileM = 256, int TileN = 128, int WarpM = 64, int TileK = 128>
struct GatedResidualEpilogueGemm {
  using ThreadblockShape = typename W4A8Tile<TileM, TileN, WarpM, TileK>::ThreadblockShape;
  using WarpShape = typename W4A8Tile<TileM, TileN, WarpM, TileK>::WarpShape;
  using ThreadblockMma = typename W4A8Tile<TileM, TileN, WarpM, TileK>::ThreadblockMma;
  static int const kPartitionsK = W4A8Tile<TileM, TileN, WarpM, TileK>::kPartitionsK;
  static int const kElementsPerAccess = 8;
  static int const kEpilogueStages = 1;

  using OutputTileThreadMap = cutlass::epilogue::threadblock::OutputTileThreadLayout<
      ThreadblockShape,
      WarpShape,
      ElementOutput,
      kElementsPerAccess,
      kEpilogueStages>;

  using Accumulator = cutlass::epilogue::threadblock::VisitorAccFetch;
  using ActivationScale = cutlass::epilogue::threadblock::VisitorColBroadcast<
      OutputTileThreadMap,
      float>;
  using Multiply = cutlass::epilogue::threadblock::VisitorCompute<
      cutlass::multiplies,
      float,
      float,
      cutlass::FloatRoundStyle::round_to_nearest>;
  using ScaleActivation = cutlass::epilogue::threadblock::Sm80EVT<
      Multiply,
      Accumulator,
      ActivationScale>;

  using WeightScale = cutlass::epilogue::threadblock::VisitorRowBroadcast<
      OutputTileThreadMap,
      float,
      cute::Stride<cute::_0, cute::_1, cute::_0>>;
  using ScaleWeight = cutlass::epilogue::threadblock::Sm80EVT<
      Multiply,
      ScaleActivation,
      WeightScale>;

  using Bias = cutlass::epilogue::threadblock::VisitorRowBroadcast<
      OutputTileThreadMap,
      ElementOutput,
      cute::Stride<cute::_0, cute::_1, cute::_0>>;
  using Add = cutlass::epilogue::threadblock::VisitorCompute<
      cutlass::plus,
      float,
      float,
      cutlass::FloatRoundStyle::round_to_nearest>;
  using AddBias = cutlass::epilogue::threadblock::Sm80EVT<
      Add,
      ScaleWeight,
      Bias>;

  using Gate = VisitorGroupedRowBroadcast<OutputTileThreadMap, ElementOutput>;
  using ApplyGate = cutlass::epilogue::threadblock::Sm80EVT<
      Multiply,
      AddBias,
      Gate>;
  using Residual = VisitorOptionalAuxLoad<
      OutputTileThreadMap,
      ElementOutput,
      cute::Stride<int64_t, cute::_1, cute::_0>>;
  using AddResidual = cutlass::epilogue::threadblock::Sm80EVT<
      Add,
      ApplyGate,
      Residual>;

  using Output = cutlass::epilogue::threadblock::VisitorAuxStore<
      OutputTileThreadMap,
      ElementOutput,
      cutlass::FloatRoundStyle::round_to_nearest,
      cute::Stride<int64_t, cute::_1, cute::_0>>;
  using OutputCallbacks = cutlass::epilogue::threadblock::Sm80EVT<Output, AddResidual>;

  // Only the default epilogue's accumulator exchange and thread map are used;
  // OutputCallbacks replaces its arithmetic and global store.
  using ThreadMapOutputOp = cutlass::epilogue::thread::LinearCombination<
      ElementOutput,
      kElementsPerAccess,
      ElementAccumulator,
      float>;
  using DefaultEpilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueTensorOp<
      ThreadblockShape,
      typename ThreadblockMma::Operator,
      kPartitionsK,
      ThreadMapOutputOp,
      kElementsPerAccess>::Epilogue;
  using Epilogue = cutlass::epilogue::threadblock::EpilogueWithVisitorCallbacks<
      DefaultEpilogue,
      OutputCallbacks,
      kEpilogueStages>;
  using Kernel = cutlass::gemm::kernel::GemmWithEpilogueVisitor<
      ThreadblockMma,
      Epilogue,
      ThreadblockSwizzle>;
  using DeviceGemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
};

template <typename scalar_t>
__device__ __forceinline__ float load_as_float(scalar_t value) {
  return static_cast<float>(value);
}

__device__ __forceinline__ float warp_max(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset));
  }
  return value;
}

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return value;
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
  float normalized = (load_as_float(value) - mean) * inv_std;
  return normalized * (1.0f + load_as_float(scale[modulation_offset + column])) +
      load_as_float(shift[modulation_offset + column]);
}

// LayerNorm (without affine parameters), AdaLN modulation, SmoothQuant
// migration and dynamic per-token A8 quantization are performed in one
// launch. No BF16 normalized/modulated activation is materialized.
template <typename scalar_t, bool kInputScaleIsInverse>
__global__ void quantize_dynamic_adaln_per_token_kernel(
    scalar_t const* __restrict__ x,
    scalar_t const* __restrict__ adaln_scale,
    scalar_t const* __restrict__ adaln_shift,
    int64_t modulation_row_stride,
    int64_t modulation_span,
    float epsilon,
    float const* __restrict__ input_scale,
    int8_t* __restrict__ quantized,
    float* __restrict__ row_scales,
    int64_t rows,
    int64_t columns) {
  int64_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  __shared__ float warp_values[16];
  __shared__ float warp_squares[16];
  __shared__ float shared_mean;
  __shared__ float shared_inv_std;
  __shared__ float shared_quant_scale;
  extern __shared__ float shared_values[];

  float local_sum = 0.0f;
  float local_square_sum = 0.0f;
  for (int64_t column = threadIdx.x; column < columns; column += blockDim.x) {
    float value = load_as_float(x[row * columns + column]);
    local_sum += value;
    local_square_sum += value * value;
  }
  local_sum = warp_sum(local_sum);
  local_square_sum = warp_sum(local_square_sum);
  if ((threadIdx.x & 31) == 0) {
    warp_values[threadIdx.x >> 5] = local_sum;
    warp_squares[threadIdx.x >> 5] = local_square_sum;
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
    shared_mean = block_sum * inverse_columns;
    float variance = fmaxf(block_square_sum * inverse_columns - shared_mean * shared_mean, 0.0f);
    shared_inv_std = rsqrtf(variance + epsilon);
  }
  __syncthreads();

  int64_t modulation_row = row / modulation_span;
  int64_t modulation_offset = modulation_row * modulation_row_stride;
  float local_max = 0.0f;
  for (int64_t column = threadIdx.x; column < columns; column += blockDim.x) {
    float value = adaln_value(
        x[row * columns + column], shared_mean, shared_inv_std,
        adaln_scale, adaln_shift, modulation_offset, column);
    if (input_scale != nullptr) {
      if constexpr (kInputScaleIsInverse) {
        value *= input_scale[column];
      } else {
        value /= input_scale[column];
      }
    }
    shared_values[column] = value;
    local_max = fmaxf(local_max, fabsf(value));
  }
  local_max = warp_max(local_max);
  if ((threadIdx.x & 31) == 0) {
    warp_values[threadIdx.x >> 5] = local_max;
  }
  __syncthreads();
  float block_max = threadIdx.x < blockDim.x / 32 ? warp_values[threadIdx.x] : 0.0f;
  if (threadIdx.x < 32) {
    block_max = warp_max(block_max);
  }
  if (threadIdx.x == 0) {
    shared_quant_scale = fmaxf(block_max, 1.0e-8f) / 127.0f;
    row_scales[row] = shared_quant_scale;
  }
  __syncthreads();

  float inverse_quant_scale = 1.0f / shared_quant_scale;
  for (int64_t column = threadIdx.x; column < columns; column += blockDim.x) {
    float value = shared_values[column];
    int quant = __float2int_rn(value * inverse_quant_scale);
    quant = max(-127, min(127, quant));
    quantized[row * columns + column] = static_cast<int8_t>(quant);
  }
}

// WAM counterpart: the activation scale is fixed per timestep, while a fixed
// semantic-stream gain is selected from the calibrated schedule. LayerNorm and
// AdaLN still happen directly in the A8 producer.
template <typename scalar_t, bool kInputScaleIsInverse>
__global__ void quantize_static_adaln_stream_kernel(
    scalar_t const* __restrict__ x,
    scalar_t const* __restrict__ adaln_scale,
    scalar_t const* __restrict__ adaln_shift,
    int64_t modulation_row_stride,
    int64_t modulation_span,
    float epsilon,
    float const* __restrict__ input_scale,
    float const* __restrict__ stream_gains,
    float const* __restrict__ activation_scale,
    int8_t* __restrict__ quantized,
    float* __restrict__ row_scales,
    int64_t rows,
    int64_t columns,
    int64_t stream_span,
    int64_t num_streams) {
  int64_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  __shared__ float warp_values[16];
  __shared__ float warp_squares[16];
  __shared__ float shared_mean;
  __shared__ float shared_inv_std;
  __shared__ float shared_gain_over_scale;

  float local_sum = 0.0f;
  float local_square_sum = 0.0f;
  for (int64_t column = threadIdx.x; column < columns; column += blockDim.x) {
    float value = load_as_float(x[row * columns + column]);
    local_sum += value;
    local_square_sum += value * value;
  }
  local_sum = warp_sum(local_sum);
  local_square_sum = warp_sum(local_square_sum);
  if ((threadIdx.x & 31) == 0) {
    warp_values[threadIdx.x >> 5] = local_sum;
    warp_squares[threadIdx.x >> 5] = local_square_sum;
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
    shared_mean = block_sum * inverse_columns;
    float variance = fmaxf(block_square_sum * inverse_columns - shared_mean * shared_mean, 0.0f);
    shared_inv_std = rsqrtf(variance + epsilon);
    int64_t stream = (row / stream_span) % num_streams;
    float gain = stream_gains[stream];
    shared_gain_over_scale = gain / activation_scale[0];
    row_scales[row] = activation_scale[0] / gain;
  }
  __syncthreads();

  int64_t modulation_row = row / modulation_span;
  int64_t modulation_offset = modulation_row * modulation_row_stride;
  for (int64_t column = threadIdx.x; column < columns; column += blockDim.x) {
    float value = adaln_value(
        x[row * columns + column], shared_mean, shared_inv_std,
        adaln_scale, adaln_shift, modulation_offset, column);
    if (input_scale != nullptr) {
      if constexpr (kInputScaleIsInverse) {
        value *= input_scale[column];
      } else {
        value /= input_scale[column];
      }
    }
    int quant = __float2int_rn(value * shared_gain_over_scale);
    quant = max(-127, min(127, quant));
    quantized[row * columns + column] = static_cast<int8_t>(quant);
  }
}

template <typename scalar_t, bool kInputScaleIsInverse>
__global__ void quantize_dynamic_per_token_kernel(
    scalar_t const* __restrict__ x,
    float const* __restrict__ input_scale,
    int8_t* __restrict__ quantized,
    float* __restrict__ row_scales,
    int64_t rows,
    int64_t columns) {
  int64_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  float local_max = 0.0f;
  for (int64_t column = threadIdx.x; column < columns; column += blockDim.x) {
    float value = load_as_float(x[row * columns + column]);
    if (input_scale != nullptr) {
      if constexpr (kInputScaleIsInverse) {
        value *= input_scale[column];
      } else {
        value /= input_scale[column];
      }
    }
    local_max = fmaxf(local_max, fabsf(value));
  }
  local_max = warp_max(local_max);
  __shared__ float warp_values[16];
  __shared__ float shared_scale;
  if ((threadIdx.x & 31) == 0) {
    warp_values[threadIdx.x >> 5] = local_max;
  }
  __syncthreads();
  float block_max = threadIdx.x < blockDim.x / 32 ? warp_values[threadIdx.x] : 0.0f;
  if (threadIdx.x < 32) {
    block_max = warp_max(block_max);
  }
  if (threadIdx.x == 0) {
    shared_scale = fmaxf(block_max, 1.0e-8f) / 127.0f;
    row_scales[row] = shared_scale;
  }
  __syncthreads();
  float inverse_scale = 1.0f / shared_scale;
  for (int64_t column = threadIdx.x; column < columns; column += blockDim.x) {
    float value = load_as_float(x[row * columns + column]);
    if (input_scale != nullptr) {
      if constexpr (kInputScaleIsInverse) {
        value *= input_scale[column];
      } else {
        value /= input_scale[column];
      }
    }
    int quant = __float2int_rn(value * inverse_scale);
    quant = max(-127, min(127, quant));
    quantized[row * columns + column] = static_cast<int8_t>(quant);
  }
}

// SmoothQuant applies a per-channel migration scale before finding each
// token's dynamic A8 scale.  Caching that transformed row in shared memory
// avoids loading x/input_scale and performing the FP32 division twice.
template <typename scalar_t, bool kInputScaleIsInverse>
__global__ void quantize_dynamic_per_token_cached_kernel(
    scalar_t const* __restrict__ x,
    float const* __restrict__ input_scale,
    int8_t* __restrict__ quantized,
    float* __restrict__ row_scales,
    int64_t rows,
    int64_t columns) {
  int64_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  extern __shared__ float transformed[];
  float local_max = 0.0f;
  for (int64_t column = threadIdx.x; column < columns; column += blockDim.x) {
    float value = load_as_float(x[row * columns + column]);
    if constexpr (kInputScaleIsInverse) {
      value *= input_scale[column];
    } else {
      value /= input_scale[column];
    }
    transformed[column] = value;
    local_max = fmaxf(local_max, fabsf(value));
  }
  local_max = warp_max(local_max);
  __shared__ float warp_values[16];
  __shared__ float shared_scale;
  if ((threadIdx.x & 31) == 0) {
    warp_values[threadIdx.x >> 5] = local_max;
  }
  __syncthreads();
  float block_max = threadIdx.x < blockDim.x / 32 ? warp_values[threadIdx.x] : 0.0f;
  if (threadIdx.x < 32) {
    block_max = warp_max(block_max);
  }
  if (threadIdx.x == 0) {
    shared_scale = fmaxf(block_max, 1.0e-8f) / 127.0f;
    row_scales[row] = shared_scale;
  }
  __syncthreads();
  float inverse_scale = 1.0f / shared_scale;
  for (int64_t column = threadIdx.x; column < columns; column += blockDim.x) {
    int quant = __float2int_rn(transformed[column] * inverse_scale);
    quant = max(-127, min(127, quant));
    quantized[row * columns + column] = static_cast<int8_t>(quant);
  }
}

template <typename scalar_t, bool kInputScaleIsInverse>
__global__ void quantize_static_per_tensor_kernel(
    scalar_t const* __restrict__ x,
    float const* __restrict__ input_scale,
    scalar_t const* __restrict__ row_gains,
    float const* __restrict__ activation_scale,
    int8_t* __restrict__ quantized,
    float* __restrict__ row_scales,
    int64_t rows,
    int64_t columns) {
  int64_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  float scale = activation_scale[0];
  float gain = load_as_float(row_gains[row]);
  if (threadIdx.x == 0) {
    row_scales[row] = scale / gain;
  }
  for (int64_t column = threadIdx.x; column < columns; column += blockDim.x) {
    float value = load_as_float(x[row * columns + column]);
    if (input_scale != nullptr) {
      if constexpr (kInputScaleIsInverse) {
        value *= input_scale[column];
      } else {
        value /= input_scale[column];
      }
    }
    int quant = __float2int_rn(value * gain / scale);
    quant = max(-127, min(127, quant));
    quantized[row * columns + column] = static_cast<int8_t>(quant);
  }
}

// WAM stores one gain per semantic stream, not one scale per token.  Mapping a
// flattened row back to its stream here avoids materializing an M-element gain
// tensor (and, more importantly, avoids the CUDA reductions and host/device
// synchronization previously used by the Python expansion path).
template <typename scalar_t, bool kInputScaleIsInverse>
__global__ void quantize_static_per_tensor_stream_kernel(
    scalar_t const* __restrict__ x,
    float const* __restrict__ input_scale,
    float const* __restrict__ stream_gains,
    float const* __restrict__ activation_scale,
    int8_t* __restrict__ quantized,
    float* __restrict__ row_scales,
    int64_t rows,
    int64_t columns,
    int64_t stream_span,
    int64_t num_streams,
    int64_t sequence_length) {
  int64_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  __shared__ float shared_gain_over_scale;
  if (threadIdx.x == 0) {
    // Negative span encodes the short proprio suffix following the text
    // prefix in each context sequence (including when the batch is >1).
    int64_t stream = stream_span > 0
        ? (row / stream_span) % num_streams
        : ((row % sequence_length) < sequence_length + stream_span ? 0 : 1);
    float scale = activation_scale[0];
    float gain = stream_gains[stream];
    shared_gain_over_scale = gain / scale;
    row_scales[row] = scale / gain;
  }
  __syncthreads();
  for (int64_t column = threadIdx.x; column < columns; column += blockDim.x) {
    float value = load_as_float(x[row * columns + column]);
    if (input_scale != nullptr) {
      if constexpr (kInputScaleIsInverse) {
        value *= input_scale[column];
      } else {
        value /= input_scale[column];
      }
    }
    int quant = __float2int_rn(value * shared_gain_over_scale);
    quant = max(-127, min(127, quant));
    quantized[row * columns + column] = static_cast<int8_t>(quant);
  }
}

void check_cutlass(cutlass::Status status, char const* operation) {
  TORCH_CHECK(
      status == cutlass::Status::kSuccess,
      operation,
      " failed: ",
      cutlassGetStatusString(status));
}

template <typename Gemm>
void launch_packed_w4a8_gemm_impl(
    torch::Tensor quantized,
    torch::Tensor packed_weight,
    typename Gemm::OutputCallbacks::Arguments const& output_callbacks,
    cudaStream_t stream) {
  using DeviceGemm = typename Gemm::DeviceGemm;
  int m = static_cast<int>(quantized.size(0));
  int k = static_cast<int>(quantized.size(1));
  int n = static_cast<int>(packed_weight.size(0));
  cutlass::gemm::GemmCoord problem_size(m, n, k);
  typename DeviceGemm::Arguments arguments(
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem_size,
      1,
      output_callbacks,
      quantized.data_ptr<int8_t>(),
      reinterpret_cast<PackedElementB const*>(packed_weight.data_ptr<uint8_t>()),
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
  DeviceGemm device_gemm;
  check_cutlass(device_gemm(arguments, nullptr, stream), "CUTLASS W4A8 launch");
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename ElementOutput, int TileM, int TileN, int WarpM = 64, int TileK = 128>
void launch_plain_w4a8_gemm(
    torch::Tensor quantized,
    torch::Tensor packed_weight,
    torch::Tensor row_scales,
    torch::Tensor weight_scales,
    std::optional<torch::Tensor> const& bias,
    torch::Tensor output,
    cudaStream_t stream) {
  int64_t n = packed_weight.size(0);
  using Gemm = PlainFusedEpilogueGemm<ElementOutput, TileM, TileN, WarpM, TileK>;
  typename Gemm::OutputCallbacks::Arguments output_callbacks{
      {
          {
              {
                  {},
                  {row_scales.data_ptr<float>(), 0.0f, {}},
                  {},
              },
              {weight_scales.data_ptr<float>(), 0.0f, {}},
              {},
          },
          {
              bias.has_value()
                  ? reinterpret_cast<ElementOutput const*>(bias->data_ptr())
                  : nullptr,
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
  launch_packed_w4a8_gemm_impl<Gemm>(quantized, packed_weight, output_callbacks, stream);
}

template <typename ElementOutput, int TileM, int TileN, int WarpM = 64, int TileK = 128>
void launch_configured_w4a8_gemm(
    torch::Tensor quantized,
    torch::Tensor packed_weight,
    torch::Tensor row_scales,
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
    launch_plain_w4a8_gemm<ElementOutput, TileM, TileN, WarpM, TileK>(
        quantized, packed_weight, row_scales, weight_scales, bias, output, stream);
    return;
  }

  TORCH_CHECK(gate.has_value(), "fused residual GEMM requires a gate tensor");
  using Gemm = GatedResidualEpilogueGemm<ElementOutput, TileM, TileN, WarpM, TileK>;
  typename Gemm::OutputCallbacks::Arguments output_callbacks{
      {
          {
              {
                {
                  {
                    {},
                    {row_scales.data_ptr<float>(), 0.0f, {}},
                    {},
                  },
                  {weight_scales.data_ptr<float>(), 0.0f, {}},
                  {},
                },
                {
                  bias.has_value()
                      ? reinterpret_cast<ElementOutput const*>(bias->data_ptr())
                      : nullptr,
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
  launch_packed_w4a8_gemm_impl<Gemm>(quantized, packed_weight, output_callbacks, stream);
}

template <typename ElementOutput>
void launch_packed_w4a8_gemm(
    torch::Tensor quantized, torch::Tensor packed_weight, torch::Tensor row_scales,
    torch::Tensor weight_scales, std::optional<torch::Tensor> const& bias,
    std::optional<torch::Tensor> const& residual, std::optional<torch::Tensor> const& gate,
    int64_t gate_span, int64_t gate_row_stride, torch::Tensor output, cudaStream_t stream) {
  // Only measured SM89 production shapes opt into the new layout by default.
  // Overrides are useful for reproducing the sweeps and are fixed before capture.
  static int const configured_tile = [] {
    auto value = std::getenv("FASTWAM_W4A8_TILE");
    if (value == nullptr || std::string(value) == "auto") return 0;
    std::string selection(value);
    TORCH_CHECK(selection == "32" || selection == "64" || selection == "128" || selection == "256",
                "FASTWAM_W4A8_TILE must be auto, 32, 64, 128, or 256");
    return std::atoi(value);
  }();
  static bool const old_small = std::getenv("FASTWAM_W4A8_EXPERIMENTAL_SMALL_TILE") != nullptr;
  int64_t m = quantized.size(0), k = quantized.size(1), n = packed_weight.size(0);
  int tile = configured_tile;
  if (tile == 0) {
    auto props = at::cuda::getCurrentDeviceProperties();
    bool measured =
        (m == 32 && ((k == 1024 && (n == 3072 || n == 4096)) ||
                     ((k == 3072 || k == 4096) && n == 1024))) ||
        (m == 129 && (k == 1024 || k == 3072) && n == 3072) ||
        (m == 294 && ((k == 3072 && (n == 3072 || n == 14336)) ||
                      (k == 14336 && n == 3072)));
    tile = measured && props->major == 8 && props->minor == 9 && !old_small ? 64 : 256;
  }
#define FASTWAM_LAUNCH_TILE(M, N, WM) \
  launch_configured_w4a8_gemm<ElementOutput, M, N, WM>( \
      quantized, packed_weight, row_scales, weight_scales, bias, residual, gate, \
      gate_span, gate_row_stride, output, stream)
  if (tile == 32) {
    FASTWAM_LAUNCH_TILE(32, 128, 32);
  } else if (tile == 64) {
    FASTWAM_LAUNCH_TILE(64, 128, 32);
  } else if (tile == 128) {
    FASTWAM_LAUNCH_TILE(128, 128, 32);
  } else if (old_small && quantized.size(0) <= 64 && !residual.has_value()) {
    launch_plain_w4a8_gemm<ElementOutput, 64, 64>(
        quantized, packed_weight, row_scales, weight_scales, bias, output, stream);
  } else {
    TORCH_CHECK(tile == 256, "FASTWAM_W4A8_TILE must be 32, 64, 128, or 256");
    FASTWAM_LAUNCH_TILE(256, 128, 64);
  }
#undef FASTWAM_LAUNCH_TILE
}

void check_inputs(
    torch::Tensor const& x,
    torch::Tensor const& packed_weight,
    torch::Tensor const& weight_scales,
    std::optional<torch::Tensor> const& bias,
    std::optional<torch::Tensor> const& input_scale) {
  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
  TORCH_CHECK(x.scalar_type() == torch::kFloat16 || x.scalar_type() == torch::kBFloat16,
              "x must be FP16 or BF16");
  TORCH_CHECK(x.dim() >= 2 && x.is_contiguous(), "x must be contiguous with at least two dimensions");
  TORCH_CHECK(packed_weight.is_cuda() && packed_weight.is_contiguous(), "packed_weight must be contiguous CUDA");
  TORCH_CHECK(packed_weight.scalar_type() == torch::kUInt8 && packed_weight.dim() == 2,
              "packed_weight must be a two-dimensional uint8 tensor");
  int64_t k = x.size(-1);
  int64_t n = packed_weight.size(0);
  TORCH_CHECK(packed_weight.size(1) * 2 == k, "packed_weight logical K does not match x");
  TORCH_CHECK(k % 32 == 0 && n % 8 == 0, "W4A8 requires K divisible by 32 and N divisible by 8");
  TORCH_CHECK(weight_scales.is_cuda() && weight_scales.scalar_type() == torch::kFloat32,
              "weight_scales must be CUDA float32");
  TORCH_CHECK(weight_scales.is_contiguous() && weight_scales.numel() == n,
              "weight_scales must be contiguous with N values");
  TORCH_CHECK(x.device() == packed_weight.device() && x.device() == weight_scales.device(),
              "all operands must be on the same CUDA device");
  if (bias.has_value()) {
    TORCH_CHECK(bias->is_cuda() && bias->is_contiguous() && bias->numel() == n,
                "bias must be contiguous CUDA with N values");
    TORCH_CHECK(bias->scalar_type() == x.scalar_type(), "bias dtype must match x");
    TORCH_CHECK(bias->device() == x.device(), "bias must be on the input device");
  }
  if (input_scale.has_value()) {
    TORCH_CHECK(input_scale->is_cuda() && input_scale->is_contiguous(),
                "input_scale must be contiguous CUDA");
    TORCH_CHECK(input_scale->scalar_type() == torch::kFloat32 && input_scale->numel() == k,
                "input_scale must be float32 with K values");
    TORCH_CHECK(input_scale->device() == x.device(), "input_scale must be on the input device");
  }
}

void check_adaln_inputs(
    torch::Tensor const& x,
    torch::Tensor const& adaln_scale,
    torch::Tensor const& adaln_shift,
    int64_t modulation_span) {
  int64_t k = x.size(-1);
  int64_t rows = x.numel() / k;
  TORCH_CHECK(adaln_scale.is_cuda() && adaln_shift.is_cuda(),
              "AdaLN scale and shift must be CUDA tensors");
  TORCH_CHECK(adaln_scale.device() == x.device() && adaln_shift.device() == x.device(),
              "AdaLN tensors must be on the input device");
  TORCH_CHECK(adaln_scale.scalar_type() == x.scalar_type() && adaln_shift.scalar_type() == x.scalar_type(),
              "AdaLN tensors must match the input dtype");
  TORCH_CHECK(adaln_scale.dim() >= 2 && adaln_shift.dim() >= 2 &&
                  adaln_scale.size(-1) == k && adaln_shift.sizes() == adaln_scale.sizes(),
              "AdaLN scale and shift must have matching [..., K] shapes");
  TORCH_CHECK(adaln_scale.stride(-1) == 1 && adaln_shift.stride(-1) == 1 &&
                  adaln_scale.stride(-2) == adaln_shift.stride(-2),
              "AdaLN scale and shift must have unit inner stride and matching row strides");
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
  TORCH_CHECK(residual.is_cuda() && residual.device() == x.device() &&
                  residual.scalar_type() == x.scalar_type() && residual.is_contiguous(),
              "residual must be contiguous CUDA with the input dtype");
  TORCH_CHECK(residual.numel() == rows * n,
              "residual must contain one value per output element");
  TORCH_CHECK(gate.is_cuda() && gate.device() == x.device() && gate.scalar_type() == x.scalar_type(),
              "gate must be CUDA with the input dtype");
  TORCH_CHECK(gate.dim() >= 2 && gate.size(-1) == n && gate.stride(-1) == 1,
              "gate must have [..., N] shape and unit inner stride");
  TORCH_CHECK(gate_span > 0 && rows == (gate.numel() / n) * gate_span,
              "gate rows and gate_span do not cover the output rows");
}

template <typename scalar_t>
torch::Tensor run_dynamic(
    torch::Tensor x,
    torch::Tensor packed_weight,
    torch::Tensor weight_scales,
    std::optional<torch::Tensor> bias,
    std::optional<torch::Tensor> input_scale,
    bool input_scale_is_inverse,
    cudaStream_t stream,
    std::optional<torch::Tensor> residual = std::nullopt,
    std::optional<torch::Tensor> gate = std::nullopt,
    int64_t gate_span = 1,
    int64_t gate_row_stride = 0) {
  int64_t k = x.size(-1);
  int64_t rows = x.numel() / k;
  int64_t n = packed_weight.size(0);
  auto quantized = torch::empty({rows, k}, x.options().dtype(torch::kInt8));
  auto row_scales = torch::empty({rows}, x.options().dtype(torch::kFloat32));
  auto output = torch::empty({rows, n}, x.options());
  int quant_threads = k >= 4096 ? 512 : 256;

  // All Cosmos-Policy SmoothQuant Linear shapes have K <= 8192, requiring at
  // most 32 KiB of shared memory per row. Keep the generic two-read path for
  // RTN and for unusually wide external callers.
  if (input_scale.has_value() && k <= 8192) {
    if (input_scale_is_inverse) {
      quantize_dynamic_per_token_cached_kernel<scalar_t, true><<<rows, quant_threads, k * sizeof(float), stream>>>(
          x.data_ptr<scalar_t>(),
          input_scale->data_ptr<float>(),
          quantized.data_ptr<int8_t>(),
          row_scales.data_ptr<float>(),
          rows,
          k);
    } else {
      quantize_dynamic_per_token_cached_kernel<scalar_t, false><<<rows, quant_threads, k * sizeof(float), stream>>>(
          x.data_ptr<scalar_t>(),
          input_scale->data_ptr<float>(),
          quantized.data_ptr<int8_t>(),
          row_scales.data_ptr<float>(),
          rows,
          k);
    }
  } else {
    if (input_scale_is_inverse) {
      quantize_dynamic_per_token_kernel<scalar_t, true><<<rows, quant_threads, 0, stream>>>(
          x.data_ptr<scalar_t>(),
          input_scale.has_value() ? input_scale->data_ptr<float>() : nullptr,
          quantized.data_ptr<int8_t>(),
          row_scales.data_ptr<float>(),
          rows,
          k);
    } else {
      quantize_dynamic_per_token_kernel<scalar_t, false><<<rows, quant_threads, 0, stream>>>(
          x.data_ptr<scalar_t>(),
          input_scale.has_value() ? input_scale->data_ptr<float>() : nullptr,
          quantized.data_ptr<int8_t>(),
          row_scales.data_ptr<float>(),
          rows,
          k);
    }
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  using ElementOutput = std::conditional_t<
      std::is_same_v<scalar_t, at::Half>,
      cutlass::half_t,
      cutlass::bfloat16_t>;
  launch_packed_w4a8_gemm<ElementOutput>(
      quantized, packed_weight, row_scales, weight_scales, bias,
      residual, gate, gate_span, gate_row_stride, output, stream);
  auto output_shape = x.sizes().vec();
  output_shape.back() = n;
  return output.view(output_shape);
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
  auto quantized = torch::empty({rows, k}, x.options().dtype(torch::kInt8));
  auto row_scales = torch::empty({rows}, x.options().dtype(torch::kFloat32));
  auto output = torch::empty({rows, n}, x.options());
  int quant_threads = k >= 4096 ? 512 : 256;
  int64_t modulation_row_stride = adaln_scale.stride(-2);
  size_t quant_shared_bytes = static_cast<size_t>(k) * sizeof(float);

  if (input_scale_is_inverse) {
    quantize_dynamic_adaln_per_token_kernel<scalar_t, true>
        <<<rows, quant_threads, quant_shared_bytes, stream>>>(
        x.data_ptr<scalar_t>(), adaln_scale.data_ptr<scalar_t>(), adaln_shift.data_ptr<scalar_t>(),
        modulation_row_stride, modulation_span, static_cast<float>(epsilon),
        input_scale.has_value() ? input_scale->data_ptr<float>() : nullptr,
        quantized.data_ptr<int8_t>(), row_scales.data_ptr<float>(), rows, k);
  } else {
    quantize_dynamic_adaln_per_token_kernel<scalar_t, false>
        <<<rows, quant_threads, quant_shared_bytes, stream>>>(
        x.data_ptr<scalar_t>(), adaln_scale.data_ptr<scalar_t>(), adaln_shift.data_ptr<scalar_t>(),
        modulation_row_stride, modulation_span, static_cast<float>(epsilon),
        input_scale.has_value() ? input_scale->data_ptr<float>() : nullptr,
        quantized.data_ptr<int8_t>(), row_scales.data_ptr<float>(), rows, k);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  using ElementOutput = std::conditional_t<
      std::is_same_v<scalar_t, at::Half>, cutlass::half_t, cutlass::bfloat16_t>;
  launch_packed_w4a8_gemm<ElementOutput>(
      quantized, packed_weight, row_scales, weight_scales, bias,
      std::nullopt, std::nullopt, 1, 0, output, stream);
  auto output_shape = x.sizes().vec();
  output_shape.back() = n;
  return output.view(output_shape);
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
  auto quantized = torch::empty({rows, k}, x.options().dtype(torch::kInt8));
  auto row_scales = torch::empty({rows}, x.options().dtype(torch::kFloat32));
  auto output = torch::empty({rows, n}, x.options());
  int quant_threads = k >= 4096 ? 512 : 256;

  if (input_scale_is_inverse) {
    quantize_static_per_tensor_kernel<scalar_t, true><<<rows, quant_threads, 0, stream>>>(
        x.data_ptr<scalar_t>(),
        input_scale.has_value() ? input_scale->data_ptr<float>() : nullptr,
        row_gains.data_ptr<scalar_t>(),
        activation_scale.data_ptr<float>(),
        quantized.data_ptr<int8_t>(),
        row_scales.data_ptr<float>(),
        rows,
        k);
  } else {
    quantize_static_per_tensor_kernel<scalar_t, false><<<rows, quant_threads, 0, stream>>>(
        x.data_ptr<scalar_t>(),
        input_scale.has_value() ? input_scale->data_ptr<float>() : nullptr,
        row_gains.data_ptr<scalar_t>(),
        activation_scale.data_ptr<float>(),
        quantized.data_ptr<int8_t>(),
        row_scales.data_ptr<float>(),
        rows,
        k);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  using ElementOutput = std::conditional_t<
      std::is_same_v<scalar_t, at::Half>,
      cutlass::half_t,
      cutlass::bfloat16_t>;
  launch_packed_w4a8_gemm<ElementOutput>(
      quantized, packed_weight, row_scales, weight_scales, bias,
      std::nullopt, std::nullopt, 1, 0, output, stream);
  auto output_shape = x.sizes().vec();
  output_shape.back() = n;
  return output.view(output_shape);
}

template <typename scalar_t>
torch::Tensor run_static_streams(
    torch::Tensor x,
    torch::Tensor packed_weight,
    torch::Tensor weight_scales,
    std::optional<torch::Tensor> bias,
    std::optional<torch::Tensor> input_scale,
    torch::Tensor activation_scale,
    torch::Tensor stream_gains,
    int64_t stream_span,
    int64_t num_streams,
    int64_t schedule_index,
    bool input_scale_is_inverse,
    cudaStream_t stream,
    std::optional<torch::Tensor> residual = std::nullopt,
    std::optional<torch::Tensor> gate = std::nullopt,
    int64_t gate_span = 1,
    int64_t gate_row_stride = 0) {
  int64_t k = x.size(-1);
  int64_t rows = x.numel() / k;
  int64_t n = packed_weight.size(0);
  auto quantized = torch::empty({rows, k}, x.options().dtype(torch::kInt8));
  auto row_scales = torch::empty({rows}, x.options().dtype(torch::kFloat32));
  auto output = torch::empty({rows, n}, x.options());
  int quant_threads = k >= 4096 ? 512 : 256;
  float const* selected_activation_scale = activation_scale.data_ptr<float>() + schedule_index;
  float const* selected_stream_gains = stream_gains.data_ptr<float>() + schedule_index * num_streams;

  if (input_scale_is_inverse) {
    quantize_static_per_tensor_stream_kernel<scalar_t, true><<<rows, quant_threads, 0, stream>>>(
        x.data_ptr<scalar_t>(), input_scale.has_value() ? input_scale->data_ptr<float>() : nullptr,
        selected_stream_gains, selected_activation_scale, quantized.data_ptr<int8_t>(),
        row_scales.data_ptr<float>(), rows, k, stream_span, num_streams, x.size(-2));
  } else {
    quantize_static_per_tensor_stream_kernel<scalar_t, false><<<rows, quant_threads, 0, stream>>>(
        x.data_ptr<scalar_t>(), input_scale.has_value() ? input_scale->data_ptr<float>() : nullptr,
        selected_stream_gains, selected_activation_scale, quantized.data_ptr<int8_t>(),
        row_scales.data_ptr<float>(), rows, k, stream_span, num_streams, x.size(-2));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  using ElementOutput = std::conditional_t<
      std::is_same_v<scalar_t, at::Half>,
      cutlass::half_t,
      cutlass::bfloat16_t>;
  launch_packed_w4a8_gemm<ElementOutput>(
      quantized, packed_weight, row_scales, weight_scales, bias,
      residual, gate, gate_span, gate_row_stride, output, stream);
  auto output_shape = x.sizes().vec();
  output_shape.back() = n;
  return output.view(output_shape);
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
    cudaStream_t stream) {
  int64_t k = x.size(-1);
  int64_t rows = x.numel() / k;
  int64_t n = packed_weight.size(0);
  auto quantized = torch::empty({rows, k}, x.options().dtype(torch::kInt8));
  auto row_scales = torch::empty({rows}, x.options().dtype(torch::kFloat32));
  auto output = torch::empty({rows, n}, x.options());
  int quant_threads = k >= 4096 ? 512 : 256;
  int64_t modulation_row_stride = adaln_scale.stride(-2);
  float const* selected_activation_scale = activation_scales.data_ptr<float>() + schedule_index;
  float const* selected_stream_gains = stream_gains.data_ptr<float>() + schedule_index * num_streams;

  if (input_scale_is_inverse) {
    quantize_static_adaln_stream_kernel<scalar_t, true><<<rows, quant_threads, 0, stream>>>(
        x.data_ptr<scalar_t>(), adaln_scale.data_ptr<scalar_t>(), adaln_shift.data_ptr<scalar_t>(),
        modulation_row_stride, modulation_span, static_cast<float>(epsilon),
        input_scale.has_value() ? input_scale->data_ptr<float>() : nullptr,
        selected_stream_gains, selected_activation_scale,
        quantized.data_ptr<int8_t>(), row_scales.data_ptr<float>(), rows, k, stream_span, num_streams);
  } else {
    quantize_static_adaln_stream_kernel<scalar_t, false><<<rows, quant_threads, 0, stream>>>(
        x.data_ptr<scalar_t>(), adaln_scale.data_ptr<scalar_t>(), adaln_shift.data_ptr<scalar_t>(),
        modulation_row_stride, modulation_span, static_cast<float>(epsilon),
        input_scale.has_value() ? input_scale->data_ptr<float>() : nullptr,
        selected_stream_gains, selected_activation_scale,
        quantized.data_ptr<int8_t>(), row_scales.data_ptr<float>(), rows, k, stream_span, num_streams);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  using ElementOutput = std::conditional_t<
      std::is_same_v<scalar_t, at::Half>, cutlass::half_t, cutlass::bfloat16_t>;
  launch_packed_w4a8_gemm<ElementOutput>(
      quantized, packed_weight, row_scales, weight_scales, bias,
      std::nullopt, std::nullopt, 1, 0, output, stream);
  auto output_shape = x.sizes().vec();
  output_shape.back() = n;
  return output.view(output_shape);
}

}  // namespace

torch::Tensor w4a8_symmetric_dynamic_cuda(
    torch::Tensor x,
    torch::Tensor packed_weight,
    torch::Tensor weight_scales,
    std::optional<torch::Tensor> bias,
    std::optional<torch::Tensor> input_scale,
    bool input_scale_is_inverse) {
  check_inputs(x, packed_weight, weight_scales, bias, input_scale);
  c10::cuda::CUDAGuard guard(x.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(x.get_device());
  if (x.scalar_type() == torch::kFloat16) {
    return run_dynamic<at::Half>(
        x, packed_weight, weight_scales, bias, input_scale, input_scale_is_inverse, stream);
  }
  return run_dynamic<at::BFloat16>(
      x, packed_weight, weight_scales, bias, input_scale, input_scale_is_inverse, stream);
}

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
    bool input_scale_is_inverse) {
  check_inputs(x, packed_weight, weight_scales, bias, input_scale);
  check_adaln_inputs(x, adaln_scale, adaln_shift, modulation_span);
  TORCH_CHECK(epsilon > 0.0, "LayerNorm epsilon must be positive");
  c10::cuda::CUDAGuard guard(x.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(x.get_device());
  if (x.scalar_type() == torch::kFloat16) {
    return run_dynamic_adaln<at::Half>(
        x, adaln_scale, adaln_shift, modulation_span, epsilon,
        packed_weight, weight_scales, bias, input_scale, input_scale_is_inverse, stream);
  }
  return run_dynamic_adaln<at::BFloat16>(
      x, adaln_scale, adaln_shift, modulation_span, epsilon,
      packed_weight, weight_scales, bias, input_scale, input_scale_is_inverse, stream);
}

torch::Tensor w4a8_symmetric_dynamic_gate_residual_cuda(
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
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(x.get_device());
  int64_t gate_row_stride = gate.stride(-2);
  if (x.scalar_type() == torch::kFloat16) {
    return run_dynamic<at::Half>(
        x, packed_weight, weight_scales, bias, input_scale, input_scale_is_inverse, stream,
        residual, gate, gate_span, gate_row_stride);
  }
  return run_dynamic<at::BFloat16>(
      x, packed_weight, weight_scales, bias, input_scale, input_scale_is_inverse, stream,
      residual, gate, gate_span, gate_row_stride);
}

torch::Tensor w4a8_symmetric_static_cuda(
    torch::Tensor x,
    torch::Tensor packed_weight,
    torch::Tensor weight_scales,
    std::optional<torch::Tensor> bias,
    std::optional<torch::Tensor> input_scale,
    torch::Tensor activation_scale,
    torch::Tensor row_gains,
    bool input_scale_is_inverse) {
  check_inputs(x, packed_weight, weight_scales, bias, input_scale);
  TORCH_CHECK(activation_scale.is_cuda() && activation_scale.scalar_type() == torch::kFloat32,
              "activation_scale must be CUDA float32");
  TORCH_CHECK(
      activation_scale.numel() == 1 && activation_scale.is_contiguous() &&
          activation_scale.device() == x.device(),
      "activation_scale must be one contiguous scalar on the input device");
  int64_t rows = x.numel() / x.size(-1);
  TORCH_CHECK(row_gains.is_cuda() && row_gains.is_contiguous(), "row_gains must be contiguous CUDA");
  TORCH_CHECK(row_gains.scalar_type() == x.scalar_type() && row_gains.numel() == rows,
              "row_gains must match x dtype and contain one value per row");
  TORCH_CHECK(row_gains.device() == x.device(), "row_gains must be on the input device");
  c10::cuda::CUDAGuard guard(x.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(x.get_device());
  if (x.scalar_type() == torch::kFloat16) {
    return run_static<at::Half>(
        x, packed_weight, weight_scales, bias, input_scale, activation_scale, row_gains,
        input_scale_is_inverse, stream);
  }
  return run_static<at::BFloat16>(
      x, packed_weight, weight_scales, bias, input_scale, activation_scale, row_gains,
      input_scale_is_inverse, stream);
}

torch::Tensor w4a8_symmetric_static_stream_cuda(
    torch::Tensor x,
    torch::Tensor packed_weight,
    torch::Tensor weight_scales,
    std::optional<torch::Tensor> bias,
    std::optional<torch::Tensor> input_scale,
    torch::Tensor activation_scale,
    torch::Tensor stream_gains,
    int64_t stream_span,
    bool input_scale_is_inverse) {
  check_inputs(x, packed_weight, weight_scales, bias, input_scale);
  TORCH_CHECK(activation_scale.is_cuda() && activation_scale.scalar_type() == torch::kFloat32,
              "activation_scale must be CUDA float32");
  TORCH_CHECK(
      activation_scale.numel() == 1 && activation_scale.is_contiguous() &&
          activation_scale.device() == x.device(),
      "activation_scale must be one contiguous scalar on the input device");
  TORCH_CHECK(stream_gains.is_cuda() && stream_gains.is_contiguous(),
              "stream_gains must be contiguous CUDA");
  TORCH_CHECK(stream_gains.scalar_type() == torch::kFloat32 && stream_gains.dim() == 1 &&
                  stream_gains.numel() > 0,
              "stream_gains must be a non-empty float32 vector");
  TORCH_CHECK(stream_gains.device() == x.device(), "stream_gains must be on the input device");
  TORCH_CHECK(stream_span > 0, "stream_span must be positive");
  int64_t rows = x.numel() / x.size(-1);
  TORCH_CHECK(rows % (stream_span * stream_gains.numel()) == 0,
              "flattened rows must contain complete semantic-stream groups");
  c10::cuda::CUDAGuard guard(x.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(x.get_device());
  if (x.scalar_type() == torch::kFloat16) {
    return run_static_streams<at::Half>(
        x, packed_weight, weight_scales, bias, input_scale, activation_scale,
        stream_gains, stream_span, stream_gains.numel(), 0, input_scale_is_inverse, stream);
  }
  return run_static_streams<at::BFloat16>(
      x, packed_weight, weight_scales, bias, input_scale, activation_scale,
      stream_gains, stream_span, stream_gains.numel(), 0, input_scale_is_inverse, stream);
}

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
    bool input_scale_is_inverse) {
  check_inputs(x, packed_weight, weight_scales, bias, input_scale);
  TORCH_CHECK(activation_scales.is_cuda() && activation_scales.scalar_type() == torch::kFloat32 &&
                  activation_scales.dim() == 1 && activation_scales.is_contiguous(),
              "activation_scales must be a contiguous CUDA float32 vector");
  TORCH_CHECK(stream_gains.is_cuda() && stream_gains.scalar_type() == torch::kFloat32 &&
                  stream_gains.dim() == 2 && stream_gains.is_contiguous(),
              "stream_gains must be a contiguous CUDA float32 schedule");
  TORCH_CHECK(activation_scales.device() == x.device() && stream_gains.device() == x.device(),
              "WAM schedules must be on the input device");
  TORCH_CHECK(stream_gains.size(0) == activation_scales.numel() && stream_gains.size(1) > 0,
              "WAM gain and activation-scale schedules do not match");
  TORCH_CHECK(schedule_index >= 0 && schedule_index < activation_scales.numel(),
              "WAM schedule index is out of range");
  int64_t num_streams = stream_gains.size(1);
  TORCH_CHECK(stream_span > 0 || (num_streams == 2 && -stream_span < x.size(-2)),
              "negative stream_span requires FastWAM text/proprio context layout");
  int64_t rows = x.numel() / x.size(-1);
  TORCH_CHECK(stream_span > 0 ? rows % (stream_span * num_streams) == 0
                                : rows % x.size(-2) == 0,
              "flattened rows must contain complete semantic-stream groups");
  c10::cuda::CUDAGuard guard(x.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(x.get_device());
  if (x.scalar_type() == torch::kFloat16) {
    return run_static_streams<at::Half>(
        x, packed_weight, weight_scales, bias, input_scale, activation_scales,
        stream_gains, stream_span, num_streams, schedule_index, input_scale_is_inverse, stream);
  }
  return run_static_streams<at::BFloat16>(
      x, packed_weight, weight_scales, bias, input_scale, activation_scales,
      stream_gains, stream_span, num_streams, schedule_index, input_scale_is_inverse, stream);
}

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
    bool input_scale_is_inverse) {
  check_inputs(x, packed_weight, weight_scales, bias, input_scale);
  check_adaln_inputs(x, adaln_scale, adaln_shift, modulation_span);
  TORCH_CHECK(epsilon > 0.0, "LayerNorm epsilon must be positive");
  TORCH_CHECK(activation_scales.is_cuda() && activation_scales.scalar_type() == torch::kFloat32 &&
                  activation_scales.dim() == 1 && activation_scales.is_contiguous(),
              "activation_scales must be a contiguous CUDA float32 vector");
  TORCH_CHECK(stream_gains.is_cuda() && stream_gains.scalar_type() == torch::kFloat32 &&
                  stream_gains.dim() == 2 && stream_gains.is_contiguous(),
              "stream_gains must be a contiguous CUDA float32 schedule");
  TORCH_CHECK(activation_scales.device() == x.device() && stream_gains.device() == x.device(),
              "WAM schedules must be on the input device");
  TORCH_CHECK(stream_gains.size(0) == activation_scales.numel() && stream_gains.size(1) > 0,
              "WAM gain and activation-scale schedules do not match");
  TORCH_CHECK(schedule_index >= 0 && schedule_index < activation_scales.numel(),
              "WAM schedule index is out of range");
  int64_t num_streams = stream_gains.size(1);
  int64_t rows = x.numel() / x.size(-1);
  TORCH_CHECK(stream_span > 0 && rows % (stream_span * num_streams) == 0,
              "flattened rows must contain complete semantic-stream groups");
  c10::cuda::CUDAGuard guard(x.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(x.get_device());
  if (x.scalar_type() == torch::kFloat16) {
    return run_static_streams_adaln<at::Half>(
        x, adaln_scale, adaln_shift, modulation_span, epsilon,
        packed_weight, weight_scales, bias, input_scale, activation_scales,
        stream_gains, stream_span, num_streams, schedule_index, input_scale_is_inverse, stream);
  }
  return run_static_streams_adaln<at::BFloat16>(
      x, adaln_scale, adaln_shift, modulation_span, epsilon,
      packed_weight, weight_scales, bias, input_scale, activation_scales,
      stream_gains, stream_span, num_streams, schedule_index, input_scale_is_inverse, stream);
}

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
    int64_t gate_span) {
  check_inputs(x, packed_weight, weight_scales, bias, input_scale);
  check_residual_gate_inputs(x, packed_weight, residual, gate, gate_span);
  TORCH_CHECK(activation_scales.is_cuda() && activation_scales.scalar_type() == torch::kFloat32 &&
                  activation_scales.dim() == 1 && activation_scales.is_contiguous(),
              "activation_scales must be a contiguous CUDA float32 vector");
  TORCH_CHECK(stream_gains.is_cuda() && stream_gains.scalar_type() == torch::kFloat32 &&
                  stream_gains.dim() == 2 && stream_gains.is_contiguous(),
              "stream_gains must be a contiguous CUDA float32 schedule");
  TORCH_CHECK(activation_scales.device() == x.device() && stream_gains.device() == x.device(),
              "WAM schedules must be on the input device");
  TORCH_CHECK(stream_gains.size(0) == activation_scales.numel() && stream_gains.size(1) > 0,
              "WAM gain and activation-scale schedules do not match");
  TORCH_CHECK(schedule_index >= 0 && schedule_index < activation_scales.numel(),
              "WAM schedule index is out of range");
  int64_t num_streams = stream_gains.size(1);
  int64_t rows = x.numel() / x.size(-1);
  TORCH_CHECK(stream_span > 0 && rows % (stream_span * num_streams) == 0,
              "flattened rows must contain complete semantic-stream groups");
  c10::cuda::CUDAGuard guard(x.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(x.get_device());
  int64_t gate_row_stride = gate.stride(-2);
  if (x.scalar_type() == torch::kFloat16) {
    return run_static_streams<at::Half>(
        x, packed_weight, weight_scales, bias, input_scale, activation_scales,
        stream_gains, stream_span, num_streams, schedule_index, input_scale_is_inverse, stream,
        residual, gate, gate_span, gate_row_stride);
  }
  return run_static_streams<at::BFloat16>(
      x, packed_weight, weight_scales, bias, input_scale, activation_scales,
      stream_gains, stream_span, num_streams, schedule_index, input_scale_is_inverse, stream,
      residual, gate, gate_span, gate_row_stride);
}
