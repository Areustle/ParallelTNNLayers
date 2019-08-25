#include "cp4Conv2dBackwardData.cuh"
#include <iostream>
#include <stdlib.h>

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

/*******************************************************************************
 * 2 Dimensional Convolution Operation using an order-4 CP decomposition.
 * Also known as a Candecomp/Parafac Decomposition, a Canonical Polyadic
 * Decomposition, and a Tensor Rank Decomposition.
 *******************************************************************************/
template<unsigned FilterDim, unsigned Rank>
__global__ void conv2d_cp4_kernel(float* __restrict__ Out,
                                  const float* __restrict__ In,
                                  const unsigned offT,
                                  const unsigned offC,
                                  const unsigned offY,
                                  const unsigned offX,
                                  const unsigned H,
                                  const unsigned W,
                                  const unsigned pad,
                                  const unsigned T,
                                  const unsigned C,
                                  const unsigned sH,
                                  const unsigned sW) {

  extern __shared__ float shared_mem[];

  const unsigned w         = threadIdx.x;
  const unsigned h         = threadIdx.y;
  const unsigned wBlockOff = blockIdx.x * blockDim.x;
  const unsigned hBlockOff = blockIdx.y * blockDim.y;
  const unsigned n         = blockIdx.z;

  float local_pixel_acc[Rank];
  for (unsigned r = 0; r < Rank; ++r) local_pixel_acc[r] = 0.0f;

  // clang-format off
  // Cooperatively load all In segment into our shared memory and pad it.
  for (unsigned t = 0; t < T; ++t){
    const float* iPtr = In + n*T*H*W + t*H*W ;
    for (unsigned hh = h; hh < sH; hh += blockDim.y)
    for (unsigned ww = w; ww < sW; ww += blockDim.x)
      shared_mem[hh*sW + ww]
          = (hh + hBlockOff >= pad       //
             && hh + hBlockOff < H + pad  //
             && ww + wBlockOff >= pad     //
             && ww + wBlockOff < W + pad)
            ? iPtr[(hh+hBlockOff-pad)*W + (ww+wBlockOff-pad)]
            : (0.0f); // Pad with Zeros if outside the bounds

    __syncthreads();

    if (hBlockOff + h >= H) continue;
    if (wBlockOff + w >= W) continue;


    float tmpxl[Rank];
    for (unsigned r = 0; r < Rank; ++r) tmpxl[r] = 0.0f;


    for (int y = 0; y < FilterDim; ++y)
    for (int x = 0; x < FilterDim; ++x)
    for (int r = 0; r < Rank; ++r){
          tmpxl[r] += shared_mem[(h+y)*sW + (w+x)]
           * const_filter[offY + (FilterDim-1-y)*Rank + r]
           * const_filter[offX + (FilterDim-1-x)*Rank + r];
    }
    for (unsigned r = 0; r < Rank; ++r)
      local_pixel_acc[r] += tmpxl[r] * const_filter[offT + t*Rank + r];
    __syncthreads();
    // clang-format on
  }

  if (hBlockOff + h >= H) return;
  if (wBlockOff + w >= W) return;

  /****************************************************************************
   * Reduce over rank while scaling by kth filter value.
   ****************************************************************************/
  for (unsigned c = 0; c < C; ++c) {

    float cth_filter_pixel = 0.0f;

    for (unsigned r = 0; r < Rank; ++r)
      cth_filter_pixel
          += local_pixel_acc[r] * const_filter[offC + c * Rank + r];

    Out[n * C * H * W + c * H * W + (h + hBlockOff) * W + w + wBlockOff]
        = cth_filter_pixel;
  }
}


float cp4_conv2d_backward_data_gpu(tensor_shape params,
                                   const float* In,
                                   const float* FT,
                                   const float* FC,
                                   const float* FY,
                                   const float* FX,
                                   float*       Out,
                                   unsigned     PROFCOUNT) {

  const unsigned N    = params.N;
  const unsigned H    = params.H;
  const unsigned W    = params.W;
  const unsigned pad  = params.pad;
  const unsigned Rank = params.Rank;
  const unsigned T    = params.T;
  const unsigned C    = params.C;
  const unsigned Y    = params.Y;
  const unsigned X    = params.X;

  if (Y != X) cerr << "Invalid filter shape. Height must equal width" << endl;

  // This implementation uses the GPU's constant memory as a fast cache to
  // hold the relatively small and unchanging filter weights. These must all
  // be accessed uniformly by the threads in a block for parallel execution.
  // Populate GPU constant memory with the 4 filters at an appropriate offset.
  const unsigned offT = 0;
  const unsigned offC = offT + (T * Rank);
  const unsigned offY = offC + (C * Rank);
  const unsigned offX = offY + (Y * Rank);
  ErrChk(cudaMemcpyToSymbol(
      const_filter, FT, sizeof(float) * (T * Rank), sizeof(float) * offT));
  ErrChk(cudaMemcpyToSymbol(
      const_filter, FC, sizeof(float) * (C * Rank), sizeof(float) * offC));
  ErrChk(cudaMemcpyToSymbol(
      const_filter, FY, sizeof(float) * (Y * Rank), sizeof(float) * offY));
  ErrChk(cudaMemcpyToSymbol(
      const_filter, FX, sizeof(float) * (X * Rank), sizeof(float) * offX));

  /* const unsigned sH   = H + 2 * pad; */
  /* const unsigned sW   = W + 2 * pad; */
  unsigned       Bh      = 4;
  unsigned       Bw      = 16;
  unsigned       sW      = X - 1 + Bw;
  unsigned       sH      = Y - 1 + Bh;
  const size_t   smsz    = sW * sH * sizeof(float);
  const unsigned WgrdDim = (W / Bw) + ((W % Bw) != 0);
  const unsigned HgrdDim = (H / Bh) + ((H % Bh) != 0);
  const dim3     Gshp(WgrdDim, HgrdDim, N);
  const dim3     Bshp(Bw, Bh, 1);

  /* conv2d_cp4_kernel<3, 4><<<Gshp, Bshp, smsz>>>( */
  /*     Out, In, offT, offC, offY, offX, H, W, pad, T, C, sH, sW); */

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float us = 0.0f;

  for (unsigned i = 0; i < PROFCOUNT; ++i) {
    ErrChk(cudaDeviceSynchronize());
    cudaEventRecord(start);
    // clang-format off
    switch (X) {
      case 1:
        switch (Rank) {
          case  1: conv2d_cp4_kernel< 1, 1><<<Gshp, Bshp, smsz>>>(Out,In,offT,offC,offY,offX,H,W,pad,T,C,sH,sW); break;
          case  2: conv2d_cp4_kernel< 1, 2><<<Gshp, Bshp, smsz>>>(Out,In,offT,offC,offY,offX,H,W,pad,T,C,sH,sW); break;
          case  4: conv2d_cp4_kernel< 1, 4><<<Gshp, Bshp, smsz>>>(Out,In,offT,offC,offY,offX,H,W,pad,T,C,sH,sW); break;
          case  8: conv2d_cp4_kernel< 1, 8><<<Gshp, Bshp, smsz>>>(Out,In,offT,offC,offY,offX,H,W,pad,T,C,sH,sW); break;
          case 16: conv2d_cp4_kernel< 1,16><<<Gshp, Bshp, smsz>>>(Out,In,offT,offC,offY,offX,H,W,pad,T,C,sH,sW); break;
          default: cerr << "Rank not supported!" << endl;
        }
        break;
      case 3:
        switch (Rank) {
          case  1: conv2d_cp4_kernel< 3, 1><<<Gshp, Bshp, smsz>>>(Out,In,offT,offC,offY,offX,H,W,pad,T,C,sH,sW); break;
          case  2: conv2d_cp4_kernel< 3, 2><<<Gshp, Bshp, smsz>>>(Out,In,offT,offC,offY,offX,H,W,pad,T,C,sH,sW); break;
          case  4: conv2d_cp4_kernel< 3, 4><<<Gshp, Bshp, smsz>>>(Out,In,offT,offC,offY,offX,H,W,pad,T,C,sH,sW); break;
          case  8: conv2d_cp4_kernel< 3, 8><<<Gshp, Bshp, smsz>>>(Out,In,offT,offC,offY,offX,H,W,pad,T,C,sH,sW); break;
          case 16: conv2d_cp4_kernel< 3,16><<<Gshp, Bshp, smsz>>>(Out,In,offT,offC,offY,offX,H,W,pad,T,C,sH,sW); break;
          default: cerr << "Rank not supported!" << endl;
        }
        break;
      case 5:
        switch (Rank) {
          case  1: conv2d_cp4_kernel< 5, 1><<<Gshp, Bshp, smsz>>>(Out,In,offT,offC,offY,offX,H,W,pad,T,C,sH,sW); break;
          case  2: conv2d_cp4_kernel< 5, 2><<<Gshp, Bshp, smsz>>>(Out,In,offT,offC,offY,offX,H,W,pad,T,C,sH,sW); break;
          case  4: conv2d_cp4_kernel< 5, 4><<<Gshp, Bshp, smsz>>>(Out,In,offT,offC,offY,offX,H,W,pad,T,C,sH,sW); break;
          case  8: conv2d_cp4_kernel< 5, 8><<<Gshp, Bshp, smsz>>>(Out,In,offT,offC,offY,offX,H,W,pad,T,C,sH,sW); break;
          case 16: conv2d_cp4_kernel< 5,16><<<Gshp, Bshp, smsz>>>(Out,In,offT,offC,offY,offX,H,W,pad,T,C,sH,sW); break;
          default: cerr << "Rank not supported!" << endl;
        }
        break;
      case 7:
        switch (Rank) {
          case  1: conv2d_cp4_kernel< 7, 1><<<Gshp, Bshp, smsz>>>(Out,In,offT,offC,offY,offX,H,W,pad,T,C,sH,sW); break;
          case  2: conv2d_cp4_kernel< 7, 2><<<Gshp, Bshp, smsz>>>(Out,In,offT,offC,offY,offX,H,W,pad,T,C,sH,sW); break;
          case  4: conv2d_cp4_kernel< 7, 4><<<Gshp, Bshp, smsz>>>(Out,In,offT,offC,offY,offX,H,W,pad,T,C,sH,sW); break;
          case  8: conv2d_cp4_kernel< 7, 8><<<Gshp, Bshp, smsz>>>(Out,In,offT,offC,offY,offX,H,W,pad,T,C,sH,sW); break;
          case 16: conv2d_cp4_kernel< 7,16><<<Gshp, Bshp, smsz>>>(Out,In,offT,offC,offY,offX,H,W,pad,T,C,sH,sW); break;
          default: cerr << "Rank not supported!" << endl;
        }
        break;
      case 9:
        switch (Rank) {
          case  1: conv2d_cp4_kernel< 9, 1><<<Gshp, Bshp, smsz>>>(Out,In,offT,offC,offY,offX,H,W,pad,T,C,sH,sW); break;
          case  2: conv2d_cp4_kernel< 9, 2><<<Gshp, Bshp, smsz>>>(Out,In,offT,offC,offY,offX,H,W,pad,T,C,sH,sW); break;
          case  4: conv2d_cp4_kernel< 9, 4><<<Gshp, Bshp, smsz>>>(Out,In,offT,offC,offY,offX,H,W,pad,T,C,sH,sW); break;
          case  8: conv2d_cp4_kernel< 9, 8><<<Gshp, Bshp, smsz>>>(Out,In,offT,offC,offY,offX,H,W,pad,T,C,sH,sW); break;
          case 16: conv2d_cp4_kernel< 9,16><<<Gshp, Bshp, smsz>>>(Out,In,offT,offC,offY,offX,H,W,pad,T,C,sH,sW); break;
          default: cerr << "Rank not supported!" << endl;
        }
        break;
      case 11:
        switch (Rank) {
          case  1: conv2d_cp4_kernel<11, 1><<<Gshp, Bshp, smsz>>>(Out,In,offT,offC,offY,offX,H,W,pad,T,C,sH,sW); break;
          case  2: conv2d_cp4_kernel<11, 2><<<Gshp, Bshp, smsz>>>(Out,In,offT,offC,offY,offX,H,W,pad,T,C,sH,sW); break;
          case  4: conv2d_cp4_kernel<11, 4><<<Gshp, Bshp, smsz>>>(Out,In,offT,offC,offY,offX,H,W,pad,T,C,sH,sW); break;
          case  8: conv2d_cp4_kernel<11, 8><<<Gshp, Bshp, smsz>>>(Out,In,offT,offC,offY,offX,H,W,pad,T,C,sH,sW); break;
          case 16: conv2d_cp4_kernel<11,16><<<Gshp, Bshp, smsz>>>(Out,In,offT,offC,offY,offX,H,W,pad,T,C,sH,sW); break;
          default: cerr << "Rank not supported!" << endl;
        }
        break;
      case 13:
        switch (Rank) {
          case  1: conv2d_cp4_kernel<13, 1><<<Gshp, Bshp, smsz>>>(Out,In,offT,offC,offY,offX,H,W,pad,T,C,sH,sW); break;
          case  2: conv2d_cp4_kernel<13, 2><<<Gshp, Bshp, smsz>>>(Out,In,offT,offC,offY,offX,H,W,pad,T,C,sH,sW); break;
          case  4: conv2d_cp4_kernel<13, 4><<<Gshp, Bshp, smsz>>>(Out,In,offT,offC,offY,offX,H,W,pad,T,C,sH,sW); break;
          case  8: conv2d_cp4_kernel<13, 8><<<Gshp, Bshp, smsz>>>(Out,In,offT,offC,offY,offX,H,W,pad,T,C,sH,sW); break;
          case 16: conv2d_cp4_kernel<13,16><<<Gshp, Bshp, smsz>>>(Out,In,offT,offC,offY,offX,H,W,pad,T,C,sH,sW); break;
          default: cerr << "Rank not supported!" << endl;
        }
        break;
      case 15:
        switch (Rank) {
          case  1: conv2d_cp4_kernel<15, 1><<<Gshp, Bshp, smsz>>>(Out,In,offT,offC,offY,offX,H,W,pad,T,C,sH,sW); break;
          case  2: conv2d_cp4_kernel<15, 2><<<Gshp, Bshp, smsz>>>(Out,In,offT,offC,offY,offX,H,W,pad,T,C,sH,sW); break;
          case  4: conv2d_cp4_kernel<15, 4><<<Gshp, Bshp, smsz>>>(Out,In,offT,offC,offY,offX,H,W,pad,T,C,sH,sW); break;
          case  8: conv2d_cp4_kernel<15, 8><<<Gshp, Bshp, smsz>>>(Out,In,offT,offC,offY,offX,H,W,pad,T,C,sH,sW); break;
          case 16: conv2d_cp4_kernel<15,16><<<Gshp, Bshp, smsz>>>(Out,In,offT,offC,offY,offX,H,W,pad,T,C,sH,sW); break;
          default: cerr << "Rank not supported!" << endl;
        }
        break;
      case 17:
        switch (Rank) {
          case  1: conv2d_cp4_kernel<17, 1><<<Gshp, Bshp, smsz>>>(Out,In,offT,offC,offY,offX,H,W,pad,T,C,sH,sW); break;
          case  2: conv2d_cp4_kernel<17, 2><<<Gshp, Bshp, smsz>>>(Out,In,offT,offC,offY,offX,H,W,pad,T,C,sH,sW); break;
          case  4: conv2d_cp4_kernel<17, 4><<<Gshp, Bshp, smsz>>>(Out,In,offT,offC,offY,offX,H,W,pad,T,C,sH,sW); break;
          case  8: conv2d_cp4_kernel<17, 8><<<Gshp, Bshp, smsz>>>(Out,In,offT,offC,offY,offX,H,W,pad,T,C,sH,sW); break;
          case 16: conv2d_cp4_kernel<17,16><<<Gshp, Bshp, smsz>>>(Out,In,offT,offC,offY,offX,H,W,pad,T,C,sH,sW); break;
          default: cerr << "Rank not supported!" << endl;
        }
        break;
      default: cerr << "Filter shape not supported!" << endl;
    }
    // clang-format on

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
