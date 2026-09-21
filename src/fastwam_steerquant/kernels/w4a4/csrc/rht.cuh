// Official had12/had28 composition; all intermediates remain on chip.
// RHT matrix provenance: pinned upstream commit (Apache-2.0); see third_party/README.md.
#include "rht_constants.cuh"

// WAM's static schedule, including unequal text/proprio streams. Unlike the
// dynamic activation quantization, match the calibration FP32->model-dtype boundary
// before D/gamma. Rotation intermediates never leave shared memory/registers.
template <typename scalar_t, int kOrder, int kBlockN>
__global__ void official_wam_quantize_kernel(
    scalar_t const* input, float const* signs, float const* inverse,
    float const* gains, float const* activation_scale, uint8_t* output,
    float* scales, int64_t span, int64_t streams, int64_t sequence_tokens) {
  constexpr int kWidth = kOrder * kBlockN;
  extern __shared__ float work[];
  int tid = threadIdx.x;
  int64_t row = blockIdx.x;
  int64_t offset = row * kWidth;
  for (int i = tid; i < kWidth; i += blockDim.x)
    work[i] = static_cast<float>(input[offset + i]) * signs[i];
  __syncthreads();
  for (int stride = 1; stride < kBlockN; stride *= 2) {
    for (int pair = tid; pair < kWidth / 2; pair += blockDim.x) {
      int a = (pair / stride) * (2 * stride) + pair % stride;
      float l = work[a], r = work[a + stride];
      work[a] = l + r;
      work[a + stride] = l - r;
    }
    __syncthreads();
  }
  int64_t stream = span < 0 ? ((row % sequence_tokens) >= sequence_tokens + span ? 1 : 0)
                            : (row / span) % streams;
  float gain = gains[stream], scale = activation_scale[0];
  if (tid == 0) scales[row] = scale / gain;
  for (int pair = tid; pair < kWidth / 2; pair += blockDim.x) {
    int i = 2 * pair, order_row = i / kBlockN, col = i % kBlockN;
    float first = 0.f, second = 0.f;
    if constexpr (kOrder == 1) {
      first = work[i]; second = work[i + 1];
    } else {
      unsigned int mask = kOrder == 12 ? kHad12Masks[order_row] : kHad28Masks[order_row];
#pragma unroll
      for (int j = 0; j < kOrder; ++j) {
        float sign = mask & (1u << j) ? 1.f : -1.f;
        first += sign * work[j * kBlockN + col];
        second += sign * work[j * kBlockN + col + 1];
      }
    }
    // Use the same two normalization factors as the reference composition.
    float norm = rsqrtf(static_cast<float>(kBlockN));
    first *= norm; second *= norm;
    if constexpr (kOrder != 1) {
      first /= sqrtf(static_cast<float>(kOrder));
      second /= sqrtf(static_cast<float>(kOrder));
    }
    first = static_cast<float>(static_cast<scalar_t>(first));
    second = static_cast<float>(static_cast<scalar_t>(second));
    int a = max(-7, min(7, __float2int_rn(first * inverse[i] * (gain / scale))));
    int b = max(-7, min(7, __float2int_rn(second * inverse[i + 1] * (gain / scale))));
    output[offset / 2 + pair] = (a & 15) | ((b & 15) << 4);
  }
}

template <typename scalar_t, int kOrder, int kBlockN>
void launch_official_wam_quantize(
    scalar_t const* input, float const* signs, float const* inverse,
    float const* gains, float const* activation_scale, uint8_t* output,
    float* scales, int64_t rows, int64_t span, int64_t streams,
    int64_t sequence_tokens, cudaStream_t stream) {
  auto kernel = &official_wam_quantize_kernel<scalar_t, kOrder, kBlockN>;
  constexpr int bytes = kOrder * kBlockN * sizeof(float);
  if constexpr (bytes > 48 * 1024)
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
  kernel<<<rows, 256, bytes, stream>>>(input, signs, inverse, gains, activation_scale,
                                     output, scales, span, streams, sequence_tokens);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t>
void dispatch_official_wam_quantize(
    scalar_t const* input, float const* signs, float const* inverse,
    float const* gains, float const* activation_scale, uint8_t* output,
    float* scales, int64_t rows, int64_t width, int64_t span, int64_t streams,
    int64_t sequence_tokens, cudaStream_t stream) {
#define WAM_ROTATE_CASE(width_, order_, block_) \
  case width_: launch_official_wam_quantize<scalar_t, order_, block_>( \
      input, signs, inverse, gains, activation_scale, output, scales, rows, span, \
      streams, sequence_tokens, stream); break
  switch (width) {
    WAM_ROTATE_CASE(256, 1, 256);
    WAM_ROTATE_CASE(512, 1, 512);
    WAM_ROTATE_CASE(1024, 1, 1024);
    WAM_ROTATE_CASE(2048, 1, 2048);
    WAM_ROTATE_CASE(3072, 12, 256);
    WAM_ROTATE_CASE(4096, 1, 4096);
    WAM_ROTATE_CASE(8192, 1, 8192);
    WAM_ROTATE_CASE(14336, 28, 512);
    default: TORCH_CHECK(false, "Unsupported official WAM rotation width: ", width);
  }
#undef WAM_ROTATE_CASE
}

template <typename scalar_t, int kOrder, int kBlockN, int kThreads = 256>
__global__ void official_rht_quantize_kernel(
    scalar_t const* input, float const* signs, uint8_t* output,
    float* scales, float clip_ratio) {
  constexpr int kWidth = kOrder * kBlockN;
  constexpr int kPairs = kWidth / (2 * kThreads);
  extern __shared__ float work[];
  __shared__ float maxima[8];
  __shared__ float scale;
  int tid = threadIdx.x;
  int64_t offset = static_cast<int64_t>(blockIdx.x) * kWidth;
  for (int i = tid; i < kWidth; i += kThreads)
    work[i] = static_cast<float>(input[offset + i]) * signs[i];
  __syncthreads();

  // Pair ownership is disjoint within each butterfly stage.
#pragma unroll
  for (int stride = 1; stride < kBlockN; stride *= 2) {
    for (int pair = tid; pair < kWidth / 2; pair += kThreads) {
      int a = (pair / stride) * (2 * stride) + pair % stride;
      float left = work[a], right = work[a + stride];
      work[a] = left + right;
      work[a + stride] = left - right;
    }
    __syncthreads();
  }

  float values[kPairs][2];
  float maximum = 0.0f;
#pragma unroll
  for (int p = 0; p < kPairs; ++p) {
    int i = 2 * (tid + p * kThreads);
    int row = i / kBlockN, column = i % kBlockN;
    unsigned int mask = kOrder == 12 ? kHad12Masks[row] : kHad28Masks[row];
    float first = 0.0f, second = 0.0f;
#pragma unroll
    for (int source = 0; source < kOrder; ++source) {
      float sign = (mask & (1u << source)) ? 1.0f : -1.0f;
      first += sign * work[source * kBlockN + column];
      second += sign * work[source * kBlockN + column + 1];
    }
    values[p][0] = first * rsqrtf(static_cast<float>(kWidth));
    values[p][1] = second * rsqrtf(static_cast<float>(kWidth));
    maximum = fmaxf(maximum, fmaxf(fabsf(values[p][0]), fabsf(values[p][1])));
  }
  maximum = warp_max(maximum);
  if ((tid & 31) == 0) maxima[tid >> 5] = maximum;
  __syncthreads();
  maximum = tid < 8 ? maxima[tid] : 0.0f;
  if (tid < 32) maximum = warp_max(maximum);
  if (tid == 0) {
    scale = fmaxf(maximum * clip_ratio, 1.0e-8f) / 7.0f;
    scales[blockIdx.x] = scale;
  }
  __syncthreads();
#pragma unroll
  for (int p = 0; p < kPairs; ++p) {
    int a = max(-7, min(7, __float2int_rn(values[p][0] / scale)));
    int b = max(-7, min(7, __float2int_rn(values[p][1] / scale)));
    output[offset / 2 + tid + p * kThreads] = (a & 15) | ((b & 15) << 4);
  }
}

template <typename scalar_t, int kOrder, int kBlockN>
void launch_official_rht_quantize(
    scalar_t const* input, float const* signs, uint8_t* output, float* scales,
    int64_t rows, float clip_ratio, cudaStream_t stream) {
  auto kernel = &official_rht_quantize_kernel<scalar_t, kOrder, kBlockN>;
  constexpr int kSharedBytes = kOrder * kBlockN * sizeof(float);
  if constexpr (kSharedBytes > 48 * 1024)
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSharedBytes));
  kernel<<<rows, 256, kSharedBytes, stream>>>(input, signs, output, scales, clip_ratio);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
