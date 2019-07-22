#include "conv.cuh"


__constant__ float const_filter[4096];

template<int TileFactor = 1>
__global__ void conv2d_full_kernel(const float* __restrict__ Input,
                                   const int pad,
                                   const int fH,
                                   const int fW,
                                   float* __restrict__ Out) {
  extern __shared__ float shared_mem[];

  // Declare useful constants. This should be cleaned up if
  // Register pressure grows too high.
  const int w         = threadIdx.x;
  const int h         = threadIdx.y;
  const int oW        = gridDim.x * blockDim.x * TileFactor;
  const int iW        = gridDim.x * blockDim.x * TileFactor + pad;
  const int hBlockOff = blockIdx.y * blockDim.y;
  const int wBlockOff = blockIdx.x * blockDim.x * TileFactor;

  // Shift the Input pointer to our Region Of Interest
  Input += hBlockOff * iW + wBlockOff;
  Out += hBlockOff * oW + wBlockOff;

  // Cooperatively load all input segment into our shared memory.
  const int jEnd = fH - 1 + blockDim.y;
  const int iEnd = fW - 1 + blockDim.x;
  const int sW   = fW - 1 + blockDim.x * TileFactor;
  // clang-format off
  for (int j = h; j < jEnd; j += blockDim.y)
  for (int i = w; i < iEnd; i += blockDim.x)
  #pragma unroll
  for (int t = 0; t < TileFactor; ++t)
    shared_mem[j*sW + i+(t*blockDim.x)] = Input[j*iW + i+(t*blockDim.x)];

  __syncthreads();

  // Build sum by tiling factor
  float sum[TileFactor];
  #pragma unroll
  for (int t = 0; t < TileFactor; ++t) sum[t] = 0.0f;

  // Perform Convolution from shared memory
  for (int r = 0; r < fH; ++r)
  for (int s = 0; s < fW; ++s)
  #pragma unroll
  for (int t = 0; t < TileFactor; ++t)
    sum[t] += shared_mem[(h+r)*sW + (w+s+(t*blockDim.x))] 
            * const_filter[r*fW + s];

  // populate output array.
  #pragma unroll
  for (int t = 0; t < TileFactor; ++t)
    Out[h*oW + w+(t*blockDim.x)] = sum[t];
  // clang-format on
}


Tensor conv2d_full_gpu(Tensor const Input, Tensor const Filter) {

  const int N  = Input.shape[0];
  const int C  = Input.shape[1];
  const int H  = Input.shape[2] - 2;
  const int W  = Input.shape[3] - 2;
  const int fH = Filter.shape[2];
  const int fW = Filter.shape[3];

  cudaMemcpyToSymbol(
      const_filter, Filter.m_data, sizeof(float) * Filter.size());

  static const int tf     = 2;
  const int        bdim   = 4; // gdim = 4, 8;
  const size_t shared_mem = fW - 1 + bdim * tf + fH - 1 + bdim * sizeof(float);
  const dim3   gridDim0(W / (tf * bdim), H / (bdim));
  const dim3   blockDim0(bdim, bdim);

  Tensor Out{ N, C, H, W };

  conv2d_full_kernel<tf><<<gridDim0, blockDim0, shared_mem>>>(
      Input.m_data, 2, fH, fW, Out.m_data);
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
