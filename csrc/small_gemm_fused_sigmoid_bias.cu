/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <ATen/cuda/EmptyTensor.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <fstream>

#include "pytorch_extension_utils.h"

namespace torch_ext {

namespace {
__device__ __forceinline__ float iTanh(float arg0) {
  const float input = static_cast<float>(arg0);
  float result;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 750) && __CUDACC_VER_MAJOR__ >= 11
  asm volatile("tanh.approx.f32 %0, %1;" : "=f"(result) : "f"(input));
#else
  result = tanhf(input);
#endif
  return result;
}

__device__ inline float sigmoid_tanh(float x) { return 0.5f * (iTanh(0.5f * x) + 1.f); }

__device__ inline float sigmoid(float x) { return 1.f / (1.f + __expf(-x)); }

template <int TN = 64, int TM = 2, int K_TILE = 64>
__launch_bounds__(TN* TM) __global__
    void gemm_fp8_smalln(const __nv_fp8_e4m3* __restrict__ A, const __nv_fp8_e4m3* __restrict__ B,
                         const __nv_bfloat16* __restrict__ bias, float* __restrict__ C, int M,
                         int K, int N, float a_scale, float b_scale) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  static_assert(K_TILE == TN, "K_TILE must equal TN");
  __shared__ float sA[TM][K_TILE];
  __shared__ float sB[K_TILE][TN];

  const int tid = threadIdx.x;
  const int t = tid / K_TILE;
  const int j = tid % TN;

  const int n0 = blockIdx.x * TN;
  const int m0 = blockIdx.y * TM;

  const int m = m0 + t;
  const int n = n0 + j;

  float acc = 0.f;

  for (int k0 = 0; k0 < K; k0 += K_TILE) {
    if (tid < TM * K_TILE) {
      int ta = tid / K_TILE;
      int la = tid % K_TILE;
      if ((m0 + ta) < M && (k0 + la) < K) {
        sA[ta][la] = float(A[(m0 + ta) * K + (k0 + la)]) * a_scale;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
        if ((k0 + K_TILE) < K)
          asm volatile("prefetch.global.L2 [%0];" ::"l"(A + (m0 + ta) * K + (k0 + K_TILE + la))
                       : "memory");
#endif
      }
    }

    if (tid < TN) {
#pragma unroll
      for (int ik = 0; ik < K_TILE; ++ik) {
        int kk = k0 + ik;
        if (kk < K && (n0 + tid) < N) {
          sB[ik][tid] = float(B[kk * N + (n0 + tid)]) * b_scale;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
          if ((k0 + K_TILE) < K)
            asm volatile("prefetch.global.L2 [%0];" ::"l"(B + (k0 + K_TILE + ik) * N + (n0 + tid))
                         : "memory");
#endif
        }
      }
    }

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
    __syncthreads();

    if (m < M && n < N) {
#pragma unroll
      for (int ik = 0; ik < K_TILE; ++ik) {
        if ((k0 + ik) < K) acc = fmaf(sA[t][ik], sB[ik][j], acc);
      }
    }
    __syncthreads();
  }

  if (m < M && n < N) {
    float bf = __bfloat162float(bias[n]);
    float val = sigmoid_tanh(acc) + bf;
    C[m * N + n] = val;
  }

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  asm volatile("griddepcontrol.launch_dependents;");
#endif
#else
  (void)A;
  (void)B;
  (void)C;
  (void)M;
  (void)K;
  (void)N;
  (void)a_scale;
  (void)b_scale;
#endif
}

template <int TN = 64, int TM = 2, int K_TILE = 64>
__launch_bounds__(TN* TM) __global__
    void gemm_bf16_smalln(const __nv_bfloat16* __restrict__ A, const __nv_bfloat16* __restrict__ B,
                          const __nv_bfloat16* __restrict__ bias, __nv_bfloat16* __restrict__ C,
                          int M, int K, int N) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  static_assert(K_TILE == TN, "K_TILE must equal TN");
  __shared__ float sA[TM][K_TILE];
  __shared__ float sB[K_TILE][TN];

  const int tid = threadIdx.x;
  const int t = tid / K_TILE;
  const int j = tid % TN;

  const int n0 = blockIdx.x * TN;
  const int m0 = blockIdx.y * TM;

  const int m = m0 + t;
  const int n = n0 + j;

  float acc = 0.f;

  for (int k0 = 0; k0 < K; k0 += K_TILE) {
    if (tid < TM * K_TILE) {
      int ta = tid / K_TILE;
      int la = tid % K_TILE;
      if ((m0 + ta) < M && (k0 + la) < K) {
        sA[ta][la] = __bfloat162float(A[(m0 + ta) * K + (k0 + la)]);
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
        if ((k0 + K_TILE) < K)
          asm volatile("prefetch.global.L2 [%0];" ::"l"(A + (m0 + ta) * K + (k0 + K_TILE + la))
                       : "memory");
#endif
      }
    }

    if (tid < TN) {
#pragma unroll
      for (int ik = 0; ik < K_TILE; ++ik) {
        int kk = k0 + ik;
        if (kk < K && (n0 + tid) < N) {
          sB[ik][tid] = __bfloat162float(B[(n0 + tid) * K + kk]);
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
          if ((k0 + K_TILE) < K)
            asm volatile("prefetch.global.L2 [%0];" ::"l"(B + (n0 + tid) * K + (k0 + K_TILE + ik))
                         : "memory");
#endif
        }
      }
    }

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
    __syncthreads();

    if (m < M && n < N) {
#pragma unroll
      for (int ik = 0; ik < K_TILE; ++ik) {
        if ((k0 + ik) < K) acc = fmaf(sA[t][ik], sB[ik][j], acc);
      }
    }
    __syncthreads();
  }

  if (m < M && n < N) {
    float bf = __bfloat162float(bias[n]);
    float val = sigmoid_tanh(acc) + bf;
    C[m * N + n] = __float2bfloat16(val);
  }

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  asm volatile("griddepcontrol.launch_dependents;");
#endif
#else
  (void)A;
  (void)B;
  (void)C;
  (void)M;
  (void)K;
  (void)N;
#endif
}

}  // namespace

at::Tensor small_gemm_fused_sigmoid_bias_impl(at::Tensor const& A, at::Tensor const& B,
                                              at::Tensor const& bias, at::Tensor out,
                                              at::Tensor workspace_buffer) {
  CHECK_INPUT_AND_TYPE(A, at::ScalarType::BFloat16);
  CHECK_INPUT_AND_TYPE(B, at::ScalarType::BFloat16);
  CHECK_INPUT_AND_TYPE(bias, at::ScalarType::BFloat16);
  CHECK_INPUT_AND_TYPE(out, at::ScalarType::BFloat16);

  int32_t M = A.size(0);
  int32_t K = A.size(1);
  int32_t N = B.size(0);
  std::ofstream fout("debug.txt", std::ios::out);
  fout << "M: " << M << ", K: " << K << ", N: " << N << std::endl;
  fout.close();

  __nv_bfloat16* dA = static_cast<__nv_bfloat16*>(A.data_ptr());
  __nv_bfloat16* dB = static_cast<__nv_bfloat16*>(B.data_ptr());
  __nv_bfloat16* dBias = static_cast<__nv_bfloat16*>(bias.data_ptr());
  __nv_bfloat16* dC = static_cast<__nv_bfloat16*>(out.data_ptr());

  dim3 block(64 * 2);
  dim3 grid((N + 64 - 1) / 64, (M + 2 - 1) / 2);
  gemm_bf16_smalln<64, 2, 64><<<grid, block>>>(dA, dB, dBias, dC, M, K, N);

  return out;
}

at::Tensor small_gemm_fused_sigmoid_bias(at::Tensor const& A, at::Tensor const& B,
                                         at::Tensor const& bias, at::Tensor out,
                                         at::Tensor workspace_buffer) {
  return small_gemm_fused_sigmoid_bias_impl(A, B, bias, out, workspace_buffer);
}

}  // namespace torch_ext

TORCH_LIBRARY_FRAGMENT(TORCH_EXTENSION_NAME, m) {
  m.def("small_gemm_fused_sigmoid_bias", &torch_ext::small_gemm_fused_sigmoid_bias);
}
