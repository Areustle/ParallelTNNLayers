#include "conv.cuh"


__constant__ float carray[4096];

__global__ void conv2d_full_kernel(const float *__restrict__ Input,
                                   const float *__restrict__ Filter,
                                   const int pad,
                                   const int R,
                                   const int S,
                                   float *__restrict__ Out) {

  /* extern __shared__ float shrd[]; */

  /* const int w = blockIdx.x; */
  /* const int W = gridDim.x; */
  /* const int h = blockIdx.y; */
}


Tensor conv2d_full_gpu(Tensor const Input, Tensor const Filter) {

  const int N = Input.shape[0];
  const int C = Input.shape[1];
  const int H = Input.shape[2] - 2;
  const int W = Input.shape[3] - 2;
  const int R = Filter.shape[2];
  const int S = Filter.shape[3];

  /* cudaMemcpyToSymbol(carray, Filter.m_data, sizeof(float) * Filter.size());
   */

  /* const dim3   gridDim0(W - 2, H - 2); */
  /* const dim3   blockDim0(1, 1); */
  /* const size_t shared_mem = 1 * sizeof(float); */

  Tensor Out{ N, C, H, W };

  /* conv2d_full_kernel<<<1, 1, shared_mem>>>( */
  /*     Input.m_data, Filter.m_data, 1, R, S, Out.m_data); */
  /* cudaDeviceSynchronize(); */

  for (int h = 0; h < H; ++h) {
    for (int w = 0; w < W; ++w) {
      float sum = 0.0f;
      for (int r = 0; r < R; ++r) {
        for (int s = 0; s < S; ++s) {
          sum += Input.m_data[(h + r) * (W + 2) + (w + s)]
                 * Filter.m_data[r * S + s];
        }
      }
      Out[h * W + w] = sum;
    }
  }

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
