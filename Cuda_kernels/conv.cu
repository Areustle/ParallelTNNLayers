#include "conv.cuh"

__constant__ float const_filter[4096];

template<int TileFactor = 1>
__global__ void conv2d_full_kernel(const float *__restrict__ Input,
                                   const int pad,
                                   const int fH,
                                   const int fW,
                                   const int C,
                                   float *__restrict__ Out) {

  extern __shared__ float shared_mem[];

  // Declare useful constants. This should be cleaned up if
  // Register pressure grows too high.
  const int w         = threadIdx.x;
  const int h         = threadIdx.y;
  const int k         = threadIdx.z * blockDim.z + blockIdx.z;
  const int Bw        = blockDim.x;
  const int Bh        = blockDim.y;
  const int oW        = gridDim.x * blockDim.x * TileFactor;
  const int oH        = gridDim.y * blockDim.y;
  const int iW        = gridDim.x * blockDim.x * TileFactor + pad;
  const int iH        = gridDim.y * blockDim.y + pad;
  const int hBlockOff = blockIdx.y * blockDim.y;
  const int wBlockOff = blockIdx.x * blockDim.x * TileFactor;

  // Shift the Input pointer to our Region Of Interest
  Input += hBlockOff * iW + wBlockOff;
  Out += hBlockOff * oW + wBlockOff;

  // Cooperatively load all input segment into our shared memory.
  const int jEnd = fH - 1 + Bh;
  const int iEnd = fW - 1 + Bw;
  const int sH   = fH - 1 + Bh;
  const int sW   = fW - 1 + Bw * TileFactor;
  // clang-format off
  for (int c = 0; c < C; ++c)
  for (int j = h; j < jEnd; j += Bh)
  for (int i = w; i < iEnd; i += Bw)
  #pragma unroll
  for (int t = 0; t < TileFactor; ++t)
    shared_mem[c*sH*sW + j*sW + i+(t*Bw)] = Input[c*iH*iW + j*iW + i+(t*Bw)];

  __syncthreads();

  // Build sum by tiling factor
  float sum[TileFactor];
#pragma unroll
  for (int t = 0; t < TileFactor; ++t) sum[t] = 0.0f;

  // Perform Convolution from shared memory
  for (int c = 0; c < C; ++c)
  for (int r = 0; r < fH; ++r)
  for (int s = 0; s < fW; ++s)
  #pragma unroll
  for (int t = 0; t < TileFactor; ++t)
    /* sum[t] += shared_mem[c*sH*sW + (h+r)*sW + (w+s+(t*Bw))] */
    /*   * const_filter[k*C*fH*fW + c*fH*fW + r*fW + s]; */
    sum[t] += Input[c*iH*iW + (h+r)*iW + (w+s+(t*Bw))]
              * const_filter[k*C*fH*fW + c*fH*fW + r*fW + s];

// populate output array.
#pragma unroll
  for (int t = 0; t < TileFactor; ++t)
    Out[k*oH*oW + h*oW + w + (t*Bw)] = sum[t];

  // clang-format on
}

Tensor conv2d_full_gpu(Tensor const Input, Tensor const Filter) {

  const int N  = Input.shape[0];
  const int C  = Input.shape[1];
  const int H  = Input.shape[2] - 2;
  const int W  = Input.shape[3] - 2;
  const int FK = Filter.shape[0];
  const int FC = Filter.shape[1];
  const int fH = Filter.shape[2];
  const int fW = Filter.shape[3];

  Tensor Out{ N, FK, H, W };

  cudaMemcpyToSymbol(
      const_filter, Filter.m_data, sizeof(float) * Filter.size());

  static const int tf   = 1;
  const int        bdim = 1;
  const size_t     shared_mem_size =
      C * (fW - 1 + bdim * tf) * (fH - 1 + bdim) * sizeof(float);
  const dim3 gridDim0(W / (tf * bdim), H / (bdim), FK);
  const dim3 blockDim0(bdim, bdim, 1);

  conv2d_full_kernel<tf><<<gridDim0, blockDim0, shared_mem_size>>>(
      Input.m_data, 2, fH, fW, C, Out.m_data);
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
