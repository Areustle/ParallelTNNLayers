#include "conv.cuh"


__constant__ float cFltr[4096];

__global__ void conv2d_full_kernel(const float *__restrict__ Input,
                                   const int pad,
                                   const int R,
                                   const int S,
                                   float *__restrict__ Out) {
  extern __shared__ float shrd[];

  // Declare useful constants. This should be cleaned up if
  // Register pressure grows too high.
  const int w         = threadIdx.x;
  const int h         = threadIdx.y;
  const int oW        = blockDim.x * gridDim.x;
  const int iW        = blockDim.x * gridDim.x + 2 * pad;
  const int wBlockOff = blockIdx.x * blockDim.x;
  const int hBlockOff = blockIdx.y * blockDim.y;

  // Shift the Input pointer to our Region Of Interest
  Input += hBlockOff * iW + wBlockOff;

  // Cooperatively load all input segment into our shared memory.
  const int sH = R - 1 + blockDim.y;
  const int sW = S - 1 + blockDim.x;

  for (int j = h; j < sH; j += blockDim.y)
    for (int i = w; i < sW; i += blockDim.x)
      shrd[j * sW + i] = Input[j * iW + i];
  __syncthreads();

  // Perform Convolution from shared memory
  float sum = 0.0f;
  for (int r = 0; r < R; ++r)
    for (int s = 0; s < S; ++s) {
      sum += shrd[(h + r) * sW + (w + s)] * cFltr[r * S + s];
    }

  Out[((hBlockOff + h) * oW) + (wBlockOff + w)] = sum;
}


Tensor conv2d_full_gpu(Tensor const Input, Tensor const Filter) {

  const int N = Input.shape[0];
  const int C = Input.shape[1];
  const int H = Input.shape[2] - 2;
  const int W = Input.shape[3] - 2;
  const int R = Filter.shape[2];
  const int S = Filter.shape[3];

  cudaMemcpyToSymbol(cFltr, Filter.m_data, sizeof(float) * Filter.size());

  const int    d          = 8;
  const size_t shared_mem = H * W * N * C * sizeof(float);
  /* const int    tile_factor = 2; */
  /* const dim3   gridDim0(W / (d * tile_factor), H / (d * tile_factor)); */
  const dim3 gridDim0(W / (d), H / (d));
  const dim3 blockDim0(d, d);

  Tensor Out{ N, C, H, W };

  conv2d_full_kernel<<<gridDim0, blockDim0, shared_mem>>>(
      Input.m_data, 1, R, S, Out.m_data);
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
