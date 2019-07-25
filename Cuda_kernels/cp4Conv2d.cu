#include "cp4Conv2d.cuh"

__constant__ float const_filter[4096];

template<int TileFactor = 1>
__global__ void conv2d_cp4_kernel(const float* __restrict__ Input,
                                  const int Rank,
                                  const int pad,
                                  const int offset_fK,
                                  const int fK,
                                  const int offset_fC,
                                  const int fC,
                                  const int offset_fH,
                                  const int fH,
                                  const int offset_fW,
                                  const int fW,
                                  const int N,
                                  const int C,
                                  float* __restrict__ Out) {

  extern __shared__ float shared_mem[];

  // Declare useful constants. This should be cleaned up if
  // Register pressure grows too high.
  const int w         = threadIdx.x;
  const int h         = threadIdx.y;
  const int k         = blockDim.z * blockIdx.z + threadIdx.z;
  const int Bw        = blockDim.x;
  const int Bh        = blockDim.y;
  const int oK        = gridDim.z * blockDim.z;
  const int oW        = gridDim.x * blockDim.x * TileFactor;
  const int oH        = gridDim.y * blockDim.y;
  const int iW        = gridDim.x * blockDim.x * TileFactor + pad;
  const int iH        = gridDim.y * blockDim.y + pad;
  const int hBlockOff = blockIdx.y * blockDim.y;
  const int wBlockOff = blockIdx.x * blockDim.x * TileFactor;
  const int jEnd      = fH - 1 + Bh;
  const int iEnd      = fW - 1 + Bw;
  const int sH        = fH - 1 + Bh;
  const int sW        = fW - 1 + Bw * TileFactor;

  for (int n = 0; n < N; ++n) {
    // Shift the Input pointer to our Region Of Interest
    const float* iPtr = Input             // Original input pointer
                        + n * C * iH * iW // batch number offset for this thread
                        + hBlockOff * iW  // h offset for this thread
                        + wBlockOff;      // w offset for this thread

    float* oPtr = Out                // Original output pointer
                  + n * oK * oH * oW // batch offset
                  + k * oH * oW      // filter offset
                  + hBlockOff * oW   // h offset
                  + wBlockOff;       // w offset
    // clang-format off

    // Cooperatively load all input segment into our shared memory.
    for (int c = 0; c < C; ++c)         // For every channel
    for (int j = h; j < jEnd; j += Bh)  // For every participating h pixel
    for (int i = w; i < iEnd; i += Bw)  // For every participating w pixel
    #pragma unroll
    for (int t = 0; t < TileFactor; ++t)
      shared_mem[c*sH*sW + j*sW + i+(t*Bw)] = iPtr[c*iH*iW + j*iW + i+(t*Bw)];

    __syncthreads();

    // Build sum by tiling factor
    float sum[TileFactor];
    #pragma unroll
    for (int t = 0; t < TileFactor; ++t) sum[t] = 0.0f;

    // Perform Convolution from shared memory
    for (int c = 0; c < C; ++c)
    for (int fh = 0; fh < fH; ++fh)
    for (int fw = 0; fw < fW; ++fw)
    for (int rr = 0; rr < Rank; ++rr)
    #pragma unroll
    for (int t = 0; t < TileFactor; ++t)
      sum[t] += shared_mem[c*sH*sW + (h+fh)*sW + (w+fw+(t*Bw))]
            *  const_filter[offset_fK + k*Rank + rr]
            *  const_filter[offset_fC + c*Rank + rr]
            *  const_filter[offset_fH + fh*Rank + rr]
            *  const_filter[offset_fW + fw*Rank + rr];
        /* * const_filter[k*C*fH*fW + c*fH*fW + fh*fW + fw]; */

    // populate output array.
    #pragma unroll
    for (int t = 0; t < TileFactor; ++t)
      oPtr[h*oW + w + (t*Bw)] = sum[t];

    // clang-format on
  }
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
  const size_t offset_fC = offset_fK + sizeof(float) * FilterK.size();
  const size_t offset_fH = offset_fC + sizeof(float) * FilterC.size();
  const size_t offset_fW = offset_fH + sizeof(float) * FilterH.size();
  cudaMemcpyToSymbol(
      const_filter + offset_fK, FilterK.m_data, sizeof(float) * FilterK.size());
  cudaMemcpyToSymbol(
      const_filter + offset_fC, FilterC.m_data, sizeof(float) * FilterC.size());
  cudaMemcpyToSymbol(
      const_filter + offset_fH, FilterH.m_data, sizeof(float) * FilterH.size());
  cudaMemcpyToSymbol(
      const_filter + offset_fW, FilterW.m_data, sizeof(float) * FilterW.size());

  // Build the kernel launch parameters
  static const int tf   = 1;                 // Tile Factor
  const int        bdim = 1;                 // Block Dimension
  const size_t     smsz = C                  // Shared Memory size for block
                      * (fW - 1 + bdim * tf) //
                      * (fH - 1 + bdim)      //
                      * sizeof(float);

  const dim3 gD(W / (tf * bdim), H / (bdim), fK / bdim); // Cuda Grid Dimension
  const dim3 bD(bdim, bdim, bdim);                       // Cuda Block Dimension

  conv2d_cp4_kernel<tf><<<gD, bD, smsz>>>(Input.m_data, //
                                          2 * pad,      //
                                          Rank,
                                          offset_fK, //
                                          fK,        //
                                          offset_fC, //
                                          fC,        //
                                          offset_fH, //
                                          fH,        //
                                          offset_fW, //
                                          fW,        //
                                          N,         //
                                          C,         //
                                          Out.m_data);
  cudaDeviceSynchronize();

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