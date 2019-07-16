#include "cp4Conv2d.h"


__constant__ float *carray[1 << 13];

__global__ void conv2d_full_kernel(const float *__restrict__ Input,
                                   const int C,
                                   const int pad,
                                   const float *__restrict__ Filter,
                                   const int K,
                                   const int R,
                                   const int S,
                                   float *__restrict__ Out) {

  extern __shared__ float shrd[];

  const int n = blockIdx.x;
  const int H = blockDim.y * gridDim.y;
  const int h = threadIdx.y + blockIdx.y * blockDim.y;
  const int W = blockDim.z * gridDim.z;
  const int w = threadIdx.z + blockIdx.z * blockDim.z;

  // clang-format off
  for (int k = 0; k < K; ++k) {
    float sum = 0.0f;
    for (int c = 0; c < C; ++c)
    for (int r = 0; r < R; ++r)
    for (int s = 0; s < S; ++s) {
      const int hIdx = h + (r - pad);
      const int wIdx = w + (s - pad);
      if (hIdx >= 0 && hIdx < H && wIdx >= 0 && wIdx < W)
        sum += Input[n*C*H*W + c*H*W + hIdx*W + wIdx]
               * Filter[k*C*R*S + c*R*S + r*S + s];
    }
    Out[n*C*H*W + k*H*W + h*W + w] = sum;
  }
  // clang-format on
}


Tensor conv2d_full_gpu(Tensor const Input, Tensor const Filter) {

  const int N   = Input.shape[0];
  const int C   = Input.shape[1];
  const int H   = Input.shape[2];
  const int W   = Input.shape[3];
  const int K   = Filter.shape[0];
  const int FC  = Filter.shape[1];
  const int R   = Filter.shape[2];
  const int S   = Filter.shape[3];
  const int pad = R / 2;

  const int    TPBD = 1;
  const dim3   gridDim0(N, H / TPBD, W / TPBD);
  const dim3   blockDim0(1, TPBD, TPBD);
  const size_t shared_mem = (2 * pad + TPBD) * (2 * pad + TPBD) * sizeof(float);

  Tensor Out{ N, C, H, W };

  conv2d_full_kernel<<<gridDim0, blockDim0, shared_mem>>>(
      Input.m_data, C, pad, Filter.m_data, K, R, S, Out.m_data);
  cudaDeviceSynchronize();
  return Out;
}


Tensor conv2d_full_cpu(Tensor const Input, Tensor const Filter) {

  const int N  = Input.shape[0];
  const int C  = Input.shape[1];
  const int H  = Input.shape[2];
  const int W  = Input.shape[3];
  const int FK = Filter.shape[0];
  const int FC = Filter.shape[1];
  const int FR = Filter.shape[2];
  const int FS = Filter.shape[3];

  const int FRCenter = FR / 2;
  const int FSCenter = FS / 2;

  Tensor Out{ N, C, H, W };

  // clang-format off
  for (int n = 0; n < N; ++n)
  for (int fk = 0; fk < FK; ++fk)
  for (int h = 0; h < H; ++h)
  for (int w = 0; w < W; ++w){
    float sum = 0.0f;
    for (int c = 0; c < C; ++c)
    for (int fr = 0; fr < FR; ++fr)
    for (int fs = 0; fs < FS; ++fs){

      const int hIdx = h + (fr - FRCenter);
      const int wIdx = w + (fs - FSCenter);

      if(hIdx >= 0 && hIdx < H && wIdx >= 0 && wIdx < W){
            sum += Input.m_data[n*C*H*W + c*H*W + hIdx*W + wIdx]
            *  Filter.m_data[fk*C*FR*FS + c*FR*FS + fr*FS + fs];
      }

    }
    Out.m_data[n*C*H*W + fk*H*W + h*W + w] = sum;
  }
  // clang-format on

  return Out;
}


Tensor conv2d_cp4_cpu(Tensor const Input,
                      Tensor const FilterK,
                      Tensor const FilterC,
                      Tensor const FilterR,
                      Tensor const FilterS) {

  const int N    = Input.shape[0];
  const int C    = Input.shape[1];
  const int H    = Input.shape[2];
  const int W    = Input.shape[3];
  const int Rank = FilterK.shape[1];
  const int FK   = FilterK.shape[0];
  const int FC   = FilterC.shape[0];
  const int FR   = FilterR.shape[0];
  const int FS   = FilterS.shape[0];

  const int FRCenter = FR / 2;
  const int FSCenter = FS / 2;

  Tensor Out{ N, C, H, W };

  // clang-format off
  for (int n = 0; n < N; ++n)
  for (int fk = 0; fk < FK; ++fk)
  for (int h = 0; h < H; ++h)
  for (int w = 0; w < W; ++w){
    float sum = 0.0f;
    for (int c = 0; c < C; ++c)
    for (int rr = 0; rr < Rank; ++rr)
    for (int fr = 0; fr < FR; ++fr)
    for (int fs = 0; fs < FS; ++fs){

      const int hIdx = h + (fr - FRCenter);
      const int wIdx = w + (fs - FSCenter);

      if(hIdx >= 0 && hIdx < H && wIdx >= 0 && wIdx < W){
            sum += Input.m_data[n*C*H*W + c*H*W + hIdx*W + wIdx]
            *  FilterK.m_data[fk*Rank + rr]
            *  FilterC.m_data[c*Rank + rr]
            *  FilterR.m_data[fr*Rank + rr]
            *  FilterS.m_data[fs*Rank + rr];
      }

    }
    Out.m_data[n*C*H*W + fk*H*W + h*W + w] = sum;
  }
  // clang-format on
  return Out;
}
