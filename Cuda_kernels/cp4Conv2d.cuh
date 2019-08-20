#ifndef CP4CONV2D_H
#define CP4CONV2D_H

#include <iostream>
#include <stdlib.h>

#include "Tensor.cuh"

using namespace std;
// Simple cuda error checking macro
#define ErrChk(ans) \
  { CudaAssert((ans), __FILE__, __LINE__); }
inline void
CudaAssert(cudaError_t code, const char* file, int line, bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(
        stderr, "CudaAssert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}

/*******************************************************************************
   Hard coded limit to size of decomposed filter of 4096 floats = 32 KB
 ******************************************************************************/
__constant__ float const_filter[4096];

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
static constexpr unsigned Bh = 16;
static constexpr unsigned Bw = 16;
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////


/*******************************************************************************
 * 2 Dimensional Convolution Operation using an order-4 CP decomposition.
 * Also known as a Candecomp/Parafac Decomposition, a Canonical Polyadic
 * Decomposition, and a Tensor Rank Decomposition.
 *******************************************************************************/
template<unsigned N,
         unsigned C,
         unsigned H,
         unsigned W,
         unsigned pad,
         unsigned fK,
         unsigned fH,
         unsigned fW,
         unsigned fRank,
         unsigned offK,
         unsigned offC,
         unsigned offH,
         unsigned offW,
         unsigned sW,
         unsigned sH>
__global__ void
conv2d_cp4_kernel(float* __restrict__ Out, const float* __restrict__ Input) {

  extern __shared__ float shared_mem[];

  const unsigned w         = threadIdx.x;
  const unsigned h         = threadIdx.y;
  const unsigned wBlockOff = blockIdx.x * blockDim.x;
  const unsigned hBlockOff = blockIdx.y * blockDim.y;
  const unsigned n         = blockIdx.z;

  float local_pixel_acc[fRank];
  for (unsigned r = 0; r < fRank; ++r) local_pixel_acc[r] = 0.0f;

  for (unsigned c = 0; c < C; ++c) {

    // Shift the Global pointers to our Region Of interest
    const float* iPtr = Input + n * C * H * W + c * H * W;

    // Cooperatively load all input segment into our shared memory and pad it.
    for (unsigned j = h; j < sH; j += blockDim.y)
      for (unsigned i = w; i < sW; i += blockDim.x)
        shared_mem[j * sW + i]
            = (j + hBlockOff >= pad       //
               && j + hBlockOff < H + pad //
               && i + wBlockOff >= pad    //
               && i + wBlockOff < W + pad)
                  ? iPtr[(j + hBlockOff - pad) * W + (i + wBlockOff - pad)]
                  : (0.0f); // Pad with Zeros if outside the bounds

    __syncthreads();

    // Handle block / input size mismatch. This occurs here and not earlier
    // So that these threads can still participate in the cooperative shared
    // Memory load.
    if (hBlockOff + h >= H) continue;
    if (wBlockOff + w >= W) continue;

    float tmpxl[fRank];

    for (unsigned r = 0; r < fRank; ++r) tmpxl[r] = 0.0f;

    for (unsigned fh = 0; fh < fH; ++fh)
      for (unsigned fw = 0; fw < fW; ++fw) {
#pragma unroll
        for (unsigned r = 0; r < fRank; ++r)
          tmpxl[r] += shared_mem[(h + fh) * sW + (w + fw)]
                      * const_filter[offH + fh * fRank + r]
                      * const_filter[offW + fw * fRank + r];
      }

    for (unsigned r = 0; r < fRank; ++r)
      local_pixel_acc[r] += tmpxl[r] * const_filter[offC + c * fRank + r];

    __syncthreads();
  }

  if (hBlockOff + h >= H) return;
  if (wBlockOff + w >= W) return;

  /****************************************************************************
   * Reduce over rank while scaling by kth filter value.
   ****************************************************************************/
  for (unsigned k = 0; k < fK; ++k) {

    float kth_filter_pixel = 0.0f;

    for (unsigned r = 0; r < fRank; ++r)
      kth_filter_pixel
          += local_pixel_acc[r] * const_filter[offK + k * fRank + r];

    Out[n * fK * H * W + k * H * W + (h + hBlockOff) * W + w + wBlockOff]
        = kth_filter_pixel;
  }
}


template<unsigned N,
         unsigned C,
         unsigned H,
         unsigned W,
         unsigned pad,
         unsigned fK,
         unsigned fH,
         unsigned fW,
         unsigned fRank>
float cp4_conv2d_forward(const float* In,
                         const float* FilterK,
                         const float* FilterC,
                         const float* FilterH,
                         const float* FilterW,
                         float*       Out,
                         unsigned     PROFCOUNT = 1) {

  static constexpr unsigned offK    = 0;
  static constexpr unsigned offC    = offK + (fK * fRank);
  static constexpr unsigned offH    = offC + (C * fRank);
  static constexpr unsigned offW    = offH + (fH * fRank);
  static constexpr unsigned sW      = fW - 1 + Bw;
  static constexpr unsigned sH      = fH - 1 + Bh;
  static constexpr size_t   smsz    = sW * sH * sizeof(float);
  static constexpr unsigned WgrdDim = (W / Bw) + ((W % Bw) != 0);
  static constexpr unsigned HgrdDim = (H / Bh) + ((H % Bh) != 0);
  static constexpr dim3     Gshp(WgrdDim, HgrdDim, N);
  static constexpr dim3     Bshp(Bw, Bh, 1);

  // This implementation uses the GPU's constant memory as a fast cache to
  // hold the relatively small and unchanging filter weights. These must all
  // be accessed uniformly by the threads in a block for parallel execution.
  // Populate GPU constant memory with the 4 filters at an appropriate offset.
  ErrChk(cudaMemcpyToSymbol(const_filter,
                            FilterK,
                            sizeof(float) * (fK * fRank),
                            sizeof(float) * offK));
  ErrChk(cudaMemcpyToSymbol(const_filter,
                            FilterC,
                            sizeof(float) * (C * fRank),
                            sizeof(float) * offC));
  ErrChk(cudaMemcpyToSymbol(const_filter,
                            FilterH,
                            sizeof(float) * (fH * fRank),
                            sizeof(float) * offH));
  ErrChk(cudaMemcpyToSymbol(const_filter,
                            FilterW,
                            sizeof(float) * (fW * fRank),
                            sizeof(float) * offW));


  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float us = 0.0f;

  for (unsigned i = 0; i < PROFCOUNT; ++i) {
    ErrChk(cudaDeviceSynchronize());
    cudaEventRecord(start);
    conv2d_cp4_kernel<N,
                      C,
                      H,
                      W,
                      pad,
                      fK,
                      fH,
                      fW,
                      fRank,
                      offK,
                      offC,
                      offH,
                      offW,
                      sW,
                      sH><<<Gshp, Bshp, smsz>>>(Out, In);

    cudaEventRecord(stop);
    ErrChk(cudaPeekAtLastError());
    ErrChk(cudaDeviceSynchronize());

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    us += milliseconds * 1e3;
  }
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return us / PROFCOUNT;
}

namespace CP {

  /****************************************************************************
   * Unified memory Tensorized call of Convolution in GPU
   * Call convolution with Tensors for testing
   ****************************************************************************/
  template<unsigned N,
           unsigned C,
           unsigned H,
           unsigned W,
           unsigned pad,
           unsigned fK,
           unsigned fH,
           unsigned fW,
           unsigned fRank>
  Tensor Conv2dForward(Tensor Input,
                       Tensor FilterK,
                       Tensor FilterC,
                       Tensor FilterH,
                       Tensor FilterW) {

    Tensor Out{ N, fK, H, W };

    cp4_conv2d_forward<N, C, H, W, pad, fK, fH, fW, fRank>(Input.m_data,
                                                           FilterK.m_data,
                                                           FilterC.m_data,
                                                           FilterH.m_data,
                                                           FilterW.m_data,
                                                           Out.m_data,
                                                           1);

    return Out;
  }


  /*******************************************************************************
   * Run_convolution operation with a profile count loop
   ******************************************************************************/
  template<unsigned N,
           unsigned C,
           unsigned H,
           unsigned W,
           unsigned pad,
           unsigned fK,
           unsigned fH,
           unsigned fW,
           unsigned fRank>
  float run_convolution(unsigned PROFCOUNT) {

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


    float us = cp4_conv2d_forward<N, C, H, W, pad, fK, fH, fW, fRank>(
        In, FilterK, FilterC, FilterH, FilterW, Out, PROFCOUNT);

    cudaFree(In);
    cudaFree(FilterK);
    cudaFree(FilterC);
    cudaFree(FilterH);
    cudaFree(FilterW);
    cudaFree(Out);

    return us;
  }
} // namespace CP


#endif /* CP4CONV2D_H */
