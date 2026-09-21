// Official RHT had12/had28; Apache-2.0.
// RHT constants provenance: upstream fake_quant/hadamard_utils.py; see third_party/README.md.
// Revision: 5008669b08c1f11f9b64d52d16fddd47ca754c5a
#pragma once
__device__ __constant__ unsigned int kHad12Masks[12] = {1u, 2955u, 1815u, 3629u, 3163u, 2231u, 367u, 733u, 1465u, 2929u, 1763u, 3525u};
__device__ __constant__ unsigned int kHad28Masks[28] = {268419071u, 185429047u, 102422639u, 204828893u, 141222331u, 14009207u, 28002029u, 55987673u, 111958961u, 223901537u, 179367619u, 90299783u, 180583181u, 92730907u, 16382u, 82979893u, 165943403u, 63484117u, 126951851u, 253887319u, 239371949u, 210341209u, 152279729u, 36156769u, 72297155u, 144577927u, 20753165u, 41489947u};
