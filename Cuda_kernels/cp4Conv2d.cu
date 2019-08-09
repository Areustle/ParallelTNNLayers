#include "cp4Conv2d.cuh"
#include <iostream>
#include <stdlib.h>

using namespace std;

__constant__ float const_filter[4096];

/*******************************************************************************
 * 2 Dimensional Convolution Operation using an order-4 CP decomposition.
 * Also known as a Candecomp/Parafac Decomposition, a Canonical Polyadic
 * Decomposition, and a Tensor Rank Decomposition.
 *******************************************************************************/
__global__ void conv2d_cp4_kernel(float* __restrict__ Out,
                                  const float* __restrict__ Input,
                                  const unsigned N,
                                  const unsigned C,
                                  const unsigned H,
                                  const unsigned W,
                                  const unsigned pad,
                                  const unsigned offset_fK,
                                  const unsigned offset_fC,
                                  const unsigned offset_fH,
                                  const unsigned offset_fW,
                                  const unsigned Rank,
                                  const unsigned fK,
                                  const unsigned fC,
                                  const unsigned fH,
                                  const unsigned fW,
                                  const unsigned WgrdDim,
                                  const unsigned Bw,
                                  const unsigned Bh,
                                  const unsigned Bc,
                                  const unsigned iEnd,
                                  const unsigned jEnd,
                                  const unsigned sW,
                                  const unsigned sH) {

  extern __shared__ float shared_mem[];

  const unsigned w         = threadIdx.x % Bw;
  const unsigned h         = threadIdx.x / Bw;
  const unsigned wBlockOff = (blockIdx.x % WgrdDim) * Bw;
  const unsigned hBlockOff = (blockIdx.x / WgrdDim) * Bh;

  // Grid Stride loop to handle overlarge batch (n) and filter (k) sizes
  for (unsigned n = blockIdx.z / fK; n < N; n += blockDim.z * gridDim.z) {
    for (unsigned k = blockIdx.z % fK; k < fK; k += blockDim.z * gridDim.z) {

      float partial_channel_sum = 0.0f;

      for (unsigned c = threadIdx.y; c < C; c += blockDim.y) {
        // Shift the Global pointers to our Region Of interest
        const float* iPtr = Input + n * C * H * W + c * H * W;
        float*       sPtr = shared_mem + threadIdx.y * sH * sW;

        // clang-format off
        // Cooperatively load all input segment into our shared memory and pad it.
        for (unsigned j = h; j < jEnd; j += Bh)  // For every participating h pixel
        for (unsigned i = w; i < iEnd; i += Bw)  // For every participating w pixel
          sPtr[j*sW + i] = (j+hBlockOff >= pad
                && j+hBlockOff < H+pad
                && i+wBlockOff >= pad
                && i+wBlockOff < W+pad)
            ?(iPtr[(j+hBlockOff-pad)*W         // Height
                  + (i+wBlockOff-pad)])  // Width
            :(0.0f); // Pad with Zeros if outside the bounds

        __syncthreads();
        // clang-format on

        // Handle block / input size mismatch. This occurs here and not earlier
        // So that these threads can still participate in the cooperative shared
        // Memory load.
        if (hBlockOff + h >= H) continue;
        if (wBlockOff + w >= W) continue;

        float pixel_sum = 0.0f;

        // Perform Convolution from shared memory.
        // Accumulate sum of products in 'pixel_sum' variable.
        for (unsigned rr = 0; rr < Rank; ++rr) {

          // Store intermediate results for each rank.
          float rank_sum = 0.0f;

          // sum of products for filter height and width.
          for (unsigned fh = 0; fh < fH; ++fh)
            for (unsigned fw = 0; fw < fW; ++fw)
              rank_sum += sPtr[(h + fh) * sW + (w + fw)]
                          * const_filter[offset_fH + fh * Rank + rr]
                          * const_filter[offset_fW + fw * Rank + rr];

          // Avoid redundant work in nested loop.
          rank_sum *= const_filter[offset_fK + k * Rank + rr]
                      * const_filter[offset_fC + c * Rank + rr];

          // accumulate pixel value for this channel.
          pixel_sum += rank_sum;
        }

        __syncthreads();
        sPtr[h * sW + w] = pixel_sum;
        __syncthreads();

        for (unsigned cc = blockDim.y / 2; cc > 0; cc >>= 1) {
          if (threadIdx.y < cc)
            shared_mem[threadIdx.y * sH * sW + h * sW + w]
                += shared_mem[(threadIdx.y + cc) * sH * sW + h * sW + w];
          __syncthreads();
        }

        partial_channel_sum += shared_mem[h * sW + w];
      }

      // populate output array.
      if (threadIdx.y == 0)
        Out[n * fK * H * W + k * H * W + (h + hBlockOff) * W + w + wBlockOff]
            = partial_channel_sum;
    }
  }
}


void cuda_conv2d_cp4_gpu(const float*   In,
                         const unsigned N,
                         const unsigned C,
                         const unsigned H,
                         const unsigned W,
                         const unsigned pad,
                         const float*   FilterK,
                         const float*   FilterC,
                         const float*   FilterH,
                         const float*   FilterW,
                         const unsigned fRank,
                         const unsigned fK,
                         const unsigned fC,
                         const unsigned fH,
                         const unsigned fW,
                         float*         Out) {

  // This implementation uses the GPU's constant memory as a fast cache to
  // hold the relatively small and unchanging filter weights. These must all
  // be accessed uniformly by the threads in a block for parallel execution.
  // Populate GPU constant memory with the 4 filters at an appropriate offset.
  const unsigned offset_fK = 0;
  const unsigned offset_fC = offset_fK + (fK * fRank);
  const unsigned offset_fH = offset_fC + (fC * fRank);
  const unsigned offset_fW = offset_fH + (fH * fRank);
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

  const unsigned Bh   = 8;
  const unsigned Bw   = 32;
  const unsigned Bc   = 2;
  const size_t   smsz = Bc            //
                      * (fW - 1 + Bw) //
                      * (fH - 1 + Bh) //
                      * sizeof(float);

  const unsigned WgrdDim = (W / Bw) + ((W % Bw) != 0);
  const unsigned HgrdDim = (H / Bh) + ((H % Bh) != 0);
  const dim3     Gshp(WgrdDim * HgrdDim, 1, fK * N);
  const dim3     Bshp(Bw * Bh, Bc, 1);
  const unsigned iEnd = fW - 1 + Bw;
  const unsigned jEnd = fH - 1 + Bh;
  const unsigned sW   = fW - 1 + Bw;
  const unsigned sH   = fH - 1 + Bh;

  conv2d_cp4_kernel<<<Gshp, Bshp, smsz>>>(Out,
                                          In,
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
                                          WgrdDim,
                                          Bw,
                                          Bh,
                                          Bc,
                                          iEnd,
                                          jEnd,
                                          sW,
                                          sH);
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


int main(int argc, char** argv) {

  unsigned N     = 1;
  unsigned C     = 16;
  unsigned H     = 32;
  unsigned W     = 32;
  unsigned pad   = 1;
  unsigned fK    = 16;
  unsigned fH    = 3;
  unsigned fW    = 3;
  unsigned fRank = 1;

  if (argc != 10)
    cerr << "Using Default shape" << endl;
  else {
    N     = atoi(argv[1]);
    C     = atoi(argv[2]);
    H     = atoi(argv[3]);
    W     = atoi(argv[4]);
    pad   = atoi(argv[5]);
    fK    = atoi(argv[6]);
    fH    = atoi(argv[7]);
    fW    = atoi(argv[8]);
    fRank = atoi(argv[9]);
  }


  float* In;
  float* Out;
  float* FilterK;
  float* FilterC;
  float* FilterW;
  float* FilterH;

  cudaMalloc(&In, N * C * H * W * sizeof(float));
  cudaMalloc(&FilterK, fK * fRank * sizeof(float));
  cudaMalloc(&FilterC, C * fRank * sizeof(float));
  cudaMalloc(&FilterH, fH * fRank * sizeof(float));
  cudaMalloc(&FilterW, fW * fRank * sizeof(float));
  cudaMalloc(&Out, N * fK * H * W * sizeof(float));

  cuda_conv2d_cp4_gpu(In,
                      N,
                      C,
                      H,
                      W,
                      pad,
                      FilterK,
                      FilterC,
                      FilterH,
                      FilterW,
                      fRank,
                      fK,
                      C,
                      fH,
                      fW,
                      Out);


  cudaFree(In);
  cudaFree(FilterK);
  cudaFree(FilterC);
  cudaFree(FilterH);
  cudaFree(FilterW);
  cudaFree(Out);
}
