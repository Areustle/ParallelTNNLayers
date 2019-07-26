#include "cp4Conv2d.cuh"

__constant__ float const_filter[4096];

template<int TileFactor = 1>
__global__ void conv2d_cp4_kernel(const float* __restrict__ Input,
                                  const int pad,
                                  const int offset_fK,
                                  const int offset_fC,
                                  const int offset_fH,
                                  const int offset_fW,
                                  const int Rank,
                                  const int fK,
                                  const int fC,
                                  const int fH,
                                  const int fW,
                                  const int N,
                                  const int C,
                                  float* __restrict__ Out) {

  extern __shared__ float shared_mem[];

  const unsigned n         = blockIdx.z / fK;
  const unsigned k         = blockIdx.z % fK;
  const unsigned w         = threadIdx.x;
  const unsigned h         = threadIdx.y;
  const unsigned Bw        = blockDim.x;
  const unsigned Bh        = blockDim.y;
  const unsigned oW        = gridDim.x * blockDim.x * TileFactor;
  const unsigned oH        = gridDim.y * blockDim.y;
  const unsigned iW        = gridDim.x * blockDim.x * TileFactor + pad;
  const unsigned iH        = gridDim.y * blockDim.y + pad;
  const unsigned hBlockOff = blockIdx.y * blockDim.y;
  const unsigned wBlockOff = blockIdx.x * blockDim.x * TileFactor;
  const unsigned jEnd      = fH - 1 + Bh;
  const unsigned iEnd      = fW - 1 + Bw;
  const unsigned sH        = fH - 1 + Bh;
  const unsigned sW        = fW - 1 + Bw * TileFactor;

  // Shift the Global pounsigneders to our Region Of unsignederest
  Input += n * C * iH * iW  // batch number offset for this thread
           + hBlockOff * iW // h offset for this thread
           + wBlockOff;     // w offset for this thread

  Out += n * fK * oH * oW // batch offset
         + k * oH * oW    // conv filter offset
         + hBlockOff * oW // h offset
         + wBlockOff;     // w offset
  // clang-format off

  // Cooperatively load all input segment unsignedo our shared memory.
  for (unsigned c = 0; c < C; ++c)         // For every channel
  for (unsigned j = h; j < jEnd; j += Bh)  // For every participating h pixel
  for (unsigned i = w; i < iEnd; i += Bw)  // For every participating w pixel
  #pragma unroll
  for (unsigned t = 0; t < TileFactor; ++t)
    shared_mem[c*sH*sW + j*sW + i+(t*Bw)] = Input[c*iH*iW + j*iW + i+(t*Bw)];

  __syncthreads();

  // Build sum by tiling factor
  float sum[TileFactor];
  #pragma unroll
  for (unsigned t = 0; t < TileFactor; ++t) sum[t] = 0.0f;

  // Perform Convolution from shared memory
  // currently expect this to have bank conflicts. Requires padding.
  for (unsigned c = 0; c < C; ++c)
  for (unsigned fh = 0; fh < fH; ++fh)
  for (unsigned fw = 0; fw < fW; ++fw)
  for (unsigned rr = 0; rr < Rank; ++rr)
  #pragma unroll
  for (unsigned t = 0; t < TileFactor; ++t){
    sum[t] += shared_mem[c*sH*sW + (h+fh)*sW + (w+fw+(t*Bw))]
          *  const_filter[offset_fK + k*Rank + rr]
          *  const_filter[offset_fC + c*Rank + rr]
          *  const_filter[offset_fH + fh*Rank + rr]
          *  const_filter[offset_fW + fw*Rank + rr];
  }

  // populate output array.
  #pragma unroll
  for (unsigned t = 0; t < TileFactor; ++t)
    Out[h*oW + w+(t*Bw)] = sum[t];

  // clang-format on
}

Tensor conv2d_cp4_gpu(Tensor const Input,
                      Tensor const FilterK,
                      Tensor const FilterC,
                      Tensor const FilterH,
                      Tensor const FilterW,
                      int          pad) {

  const int N    = Input.shape[0];
  const int C    = Input.shape[1];
  const int H    = Input.shape[2] - 2 * pad;
  const int W    = Input.shape[3] - 2 * pad;
  const int Rank = FilterK.shape[1];
  const int fK   = FilterK.shape[0];
  const int fC   = FilterC.shape[0];
  const int fH   = FilterH.shape[0];
  const int fW   = FilterW.shape[0];

  Tensor Out{ N, fK, H, W };

  // Populate GPU constant memory with the 4 filters at an appropriate offset.
  const size_t offset_fK = 0;
  const size_t offset_fC = offset_fK + FilterK.size();
  const size_t offset_fH = offset_fC + FilterC.size();
  const size_t offset_fW = offset_fH + FilterH.size();
  cudaMemcpyToSymbol(const_filter,
                     FilterK.m_data,
                     sizeof(float) * FilterK.size(),
                     sizeof(float) * offset_fK);
  cudaMemcpyToSymbol(const_filter,
                     FilterC.m_data,
                     sizeof(float) * FilterC.size(),
                     sizeof(float) * offset_fC);
  cudaMemcpyToSymbol(const_filter,
                     FilterH.m_data,
                     sizeof(float) * FilterH.size(),
                     sizeof(float) * offset_fH);
  cudaMemcpyToSymbol(const_filter,
                     FilterW.m_data,
                     sizeof(float) * FilterW.size(),
                     sizeof(float) * offset_fW);

  // Build the kernel launch parameters
  static const int tf   = 2;                 // Tile Factor
  const int        bdim = 16;                 // Block Dimension
  const size_t     smsz = C                  // Shared Memory size for block
                      * (fW - 1 + bdim * tf) //
                      * (fH - 1 + bdim)      //
                      * sizeof(float);

  const dim3 gD(W / (tf * bdim), H / (bdim), fK * N); // Cuda Grid Dimension
  const dim3 bD(bdim, bdim, 1);                       // Cuda Block Dimension

  conv2d_cp4_kernel<tf><<<gD, bD, smsz>>>(Input.m_data, //
                                          2 * pad,      //
                                          offset_fK,    //
                                          offset_fC,    //
                                          offset_fH,    //
                                          offset_fW,    //
                                          Rank,
                                          fK, //
                                          fC, //
                                          fH, //
                                          fW, //
                                          N,  //
                                          C,  //
                                          Out.m_data);
  cudaDeviceSynchronize();

  return Out;
}

Tensor conv2d_cp4_cpu(Tensor const Input,
                      Tensor const FilterK,
                      Tensor const FilterC,
                      Tensor const FilterR,
                      Tensor const FilterS,
                      int          pad) {

  const int N    = Input.shape[0];
  const int C    = Input.shape[1];
  const int iH   = Input.shape[2];
  const int oH   = iH - 2 * pad;
  const int iW   = Input.shape[3];
  const int oW   = iW - 2 * pad;
  const int Rank = FilterK.shape[1];
  const int fK   = FilterK.shape[0];
  const int fC   = FilterC.shape[0];
  const int fH   = FilterR.shape[0];
  const int fW   = FilterS.shape[0];

  Tensor Out{ N, C, oH, oW };

  // clang-format off
  for (int n = 0; n < N; ++n)
  for (int k = 0; k < fK; ++k)
  for (int h = 0; h < oH; ++h)
  for (int w = 0; w < oW; ++w){
    float sum = 0.0f;
    for (int c = 0; c < C; ++c)
    for (int rr = 0; rr < Rank; ++rr)
    for (int fh = 0; fh < fH; ++fh)
    for (int fw = 0; fw < fW; ++fw){
      sum += Input.m_data[n*C*iH*iW + c*iH*iW + (h+fh)*iW + w+fw]
      *  FilterK.m_data[k*Rank + rr]
      *  FilterC.m_data[c*Rank + rr]
      *  FilterR.m_data[fh*Rank + rr]
      *  FilterS.m_data[fw*Rank + rr];
    }
    Out.m_data[n*C*oH*oW + k*oH*oW + h*oW + w] = sum;
  }
  // clang-format on
  return Out;
}
