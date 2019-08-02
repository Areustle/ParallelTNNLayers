#include "cp4Conv2d.cuh"

__constant__ float const_filter[4096];

template<int TileFactor = 1>
__global__ void conv2d_cp4_kernel(const float* __restrict__ Input,
                                  const int N,
                                  const int C,
                                  const int H,
                                  const int W,
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
                                  float* __restrict__ Out) {

  extern __shared__ float shared_mem[];

  const unsigned w         = threadIdx.x;
  const unsigned h         = threadIdx.y;
  const unsigned Bw        = blockDim.x;
  const unsigned Bh        = blockDim.y;
  const unsigned hBlockOff = blockIdx.y * blockDim.y;
  const unsigned wBlockOff = blockIdx.x * blockDim.x * TileFactor;
  const unsigned jEnd      = fH - 1 + Bh;
  const unsigned iEnd      = fW - 1 + Bw;
  const unsigned sH        = fH - 1 + Bh;
  const unsigned sW        = fW - 1 + Bw * TileFactor;

  // Grid Stride loop to handle overlarge batch (n) and filter (k) sizes
  for (int n = blockIdx.z / fK; n < N; n += blockDim.z * gridDim.z) {
    for (int k = blockIdx.z % fK; k < fK; k += blockDim.z * gridDim.z) {

      // Shift the Global pointers to our Region Of interest
      const float* iPtr =
          Input + n * C * H * W; // batch number offset for this thread

      float* oPtr = Out + n * fK * H * W // batch offset
                    + k * H * W          // conv filter offset
                    + hBlockOff * W      // h offset
                    + wBlockOff;         // w offset

      // clang-format off

      // Cooperatively load all input segment into our shared memory and pad it.
      for (unsigned c = 0; c < C; ++c)         // For every channel
      for (unsigned j = h; j < jEnd; j += Bh)  // For every participating h pixel
      for (unsigned i = w; i < iEnd; i += Bw)  // For every participating w pixel
      #pragma unroll
      for (unsigned t = 0; t < TileFactor; ++t)
        shared_mem[c*sH*sW + j*sW + i+(t*Bw)]
          = (j+hBlockOff >= pad
              && j+hBlockOff < H+pad
              && i+wBlockOff >= pad
              && i+wBlockOff+(t*Bw) < W+pad)
          ?(iPtr[c*H*W                          // Channel
                  + (j+hBlockOff-pad)*W         // Height
                  + (i+wBlockOff-pad)+(t*Bw)])  // Width
          :(0.0f); // Pad with Zeros if outside the bounds

      __syncthreads();

      // Handle block / input size mismatch. This occurs here and not earlier
      // So that these threads can still participate in the cooperative shared
      // Memory load.
      if (hBlockOff + h >= H) continue;
      if (wBlockOff + w >= W) continue;

      // Build sum by tiling factor. If tiling factor is >1 then each thread
      // will calculate multiple output pixels in local registers.
      float sum[TileFactor];
      #pragma unroll
      for (unsigned t = 0; t < TileFactor; ++t) sum[t] = 0.0f;

      // Perform Convolution from shared memory.
      // Accumulate sum of products in 'sum' variable for each t.
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
        oPtr[h*W + w+(t*Bw)] = sum[t];

      // clang-format on
    }
  }
}


void cuda_conv2d_cp4_gpu(const float* In,
                         const int    N,
                         const int    C,
                         const int    H,
                         const int    W,
                         const int    pad,
                         const float* FilterK,
                         const float* FilterC,
                         const float* FilterH,
                         const float* FilterW,
                         const int    fRank,
                         const int    fK,
                         const int    fC,
                         const int    fH,
                         const int    fW,
                         float*       Out) {

  // This implementation uses the GPU's constant memory as a fast cache to
  // hold the relatively small and unchanging filter weights. These must all
  // be accessed uniformly by the threads in a block for parallel execution.
  // Populate GPU constant memory with the 4 filters at an appropriate offset.
  const size_t offset_fK = 0;
  const size_t offset_fC = offset_fK + (fK * fRank);
  const size_t offset_fH = offset_fC + (fC * fRank);
  const size_t offset_fW = offset_fH + (fH * fRank);
  cudaMemcpyToSymbol(const_filter,
                     FilterK,
                     sizeof(float) * (fK * fRank),
                     sizeof(float) * offset_fK);
  cudaMemcpyToSymbol(const_filter,
                     FilterC,
                     sizeof(float) * (fC * fRank),
                     sizeof(float) * offset_fC);
  cudaMemcpyToSymbol(const_filter,
                     FilterH,
                     sizeof(float) * (fH * fRank),
                     sizeof(float) * offset_fH);
  cudaMemcpyToSymbol(const_filter,
                     FilterW,
                     sizeof(float) * (fW * fRank),
                     sizeof(float) * offset_fW);

  const unsigned        hdim = 8;
  const unsigned        wdim = 32;
  static const unsigned tf   = 1;
  const size_t          smsz =
      C //
        // If using shared memory padding to avoid bank conflicts
      /* * ((((fW - 1) / 32) + ((fW - 1) % 32 != 0)) * 32 + bdim * tf) // */
      * (fH - 1 + wdim * tf) //
      * (fH - 1 + hdim)      //
      * sizeof(float);

  const unsigned WgrdDim = (W / (wdim * tf)) + ((W % (wdim * tf)) != 0);
  const unsigned HgrdDim = (H / hdim) + ((H % hdim) != 0);
  const dim3     Gshp(WgrdDim, HgrdDim, fK * N);
  const dim3     Bshp(wdim, hdim, 1);

  conv2d_cp4_kernel<tf><<<Gshp, Bshp, smsz>>>(In,
                                              N,
                                              C,
                                              H,
                                              W,
                                              pad,
                                              offset_fK,
                                              offset_fC,
                                              offset_fH,
                                              offset_fW,
                                              fRank,
                                              fK,
                                              fC,
                                              fH,
                                              fW,
                                              Out);
  cudaDeviceSynchronize();
}


Tensor conv2d_cp4_gpu(Tensor const Input,
                      Tensor const FilterK,
                      Tensor const FilterC,
                      Tensor const FilterH,
                      Tensor const FilterW,
                      int          pad) {

  const int N     = Input.shape[0];
  const int C     = Input.shape[1];
  const int H     = Input.shape[2];
  const int W     = Input.shape[3];
  const int fRank = FilterK.shape[1];
  const int fK    = FilterK.shape[0];
  const int fC    = FilterC.shape[0];
  const int fH    = FilterH.shape[0];
  const int fW    = FilterW.shape[0];

  Tensor Out{ N, fK, H, W };
  cuda_conv2d_cp4_gpu(Input.m_data,
                      N,
                      C,
                      H,
                      W,
                      pad,
                      FilterK.m_data,
                      FilterC.m_data,
                      FilterH.m_data,
                      FilterW.m_data,
                      fRank,
                      fK,
                      fC,
                      fH,
                      fW,
                      Out.m_data);

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
