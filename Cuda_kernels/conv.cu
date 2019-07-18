#include "conv.cuh"


__constant__ float carray[3000];

__global__ void conv2d_full_kernel(const float *__restrict__ Input,
                                   const int pad,
                                   const int R,
                                   const int S,
                                   float *__restrict__ Out) {

  /* extern __shared__ float shrd[]; */

  const int H = blockDim.y * gridDim.y;
  const int h = threadIdx.y + blockIdx.y * blockDim.y;
  const int W = blockDim.z * gridDim.z;
  const int w = threadIdx.z + blockIdx.z * blockDim.z;

  // clang-format off
  float sum = 0.0f;
  for (int r = 0; r < R; ++r)
  for (int s = 0; s < S; ++s) {
    const int hIdx = h + (r - pad);
    const int wIdx = w + (s - pad);
    if (hIdx >= 0 && hIdx < H && wIdx >= 0 && wIdx < W)
      sum += Input[hIdx*W + wIdx] * carray[r*S + s];
  }
  Out[h*W + w] = sum;
  // clang-format on
}


Tensor conv2d_full_gpu(Tensor const Input, Tensor const Filter) {

  const int N   = Input.shape[0];
  const int C   = Input.shape[1];
  const int H   = Input.shape[2];
  const int W   = Input.shape[3];
  const int R   = Filter.shape[2];
  const int S   = Filter.shape[3];
  const int pad = R / 2;

  cudaMemcpyToSymbol(carray, Filter.m_data, sizeof(float) * Filter.size());

  const int    TPBD = 1;
  const dim3   gridDim0(N, H / TPBD, W / TPBD);
  const dim3   blockDim0(1, TPBD, TPBD);
  const size_t shared_mem = (2 * pad + TPBD) * (2 * pad + TPBD) * sizeof(float);

  Tensor Out{ N, C, H, W };

  conv2d_full_kernel<<<gridDim0, blockDim0, shared_mem>>>(
      Input.m_data, pad, R, S, Out.m_data);
  cudaDeviceSynchronize();
  return Out;
}

/* // clang-format off */
/* for (int k = 0; k < K; ++k) { */
/*   float sum = 0.0f; */
/*   for (int c = 0; c < C; ++c) */
/*   for (int r = 0; r < R; ++r) */
/*   for (int s = 0; s < S; ++s) { */
/*     const int hIdx = h + (r - pad); */
/*     const int wIdx = w + (s - pad); */
/*     if (hIdx >= 0 && hIdx < H && wIdx >= 0 && wIdx < W) */
/*       sum += Input[n*C*H*W + c*H*W + hIdx*W + wIdx] */
/*              * Filter[k*C*R*S + c*R*S + r*S + s]; */
/*   } */
/*   Out[n*C*H*W + k*H*W + h*W + w] = jum; */
/* } */
/* // clang-format on */
