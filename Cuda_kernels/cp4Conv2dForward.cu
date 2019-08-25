#include "cp4Conv2dForward.cuh"
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
                                  const float* __restrict__ Input,
                                  const unsigned N,
                                  const unsigned C,
                                  const unsigned H,
                                  const unsigned W,
                                  const unsigned pad,
                                  const unsigned offT,
                                  const unsigned offC,
                                  const unsigned offH,
                                  const unsigned offW,
                                  const unsigned T,
                                  const unsigned sW,
                                  const unsigned sH) {

  extern __shared__ float shared_mem[];

  const unsigned w         = threadIdx.x;
  const unsigned h         = threadIdx.y;
  const unsigned wBlockOff = blockIdx.x * blockDim.x;
  const unsigned hBlockOff = blockIdx.y * blockDim.y;
  const unsigned n         = blockIdx.z;

  float intermediate_c_acc[Rank];
  for (unsigned r = 0; r < Rank; ++r) intermediate_c_acc[r] = 0.0f;

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

    float intermediate_pix_acc[Rank];
    for (unsigned r = 0; r < Rank; ++r) intermediate_pix_acc[r] = 0.0f;


    for (unsigned y = 0; y < FilterDim; ++y)
      for (unsigned x = 0; x < FilterDim; ++x) {
#pragma unroll
        for (unsigned r = 0; r < Rank; ++r)
          intermediate_pix_acc[r] += shared_mem[(h + y) * sW + (w + x)]
                                     * const_filter[offH + y * Rank + r]
                                     * const_filter[offW + x * Rank + r];
      }
    for (unsigned r = 0; r < Rank; ++r)
      intermediate_c_acc[r]
          += intermediate_pix_acc[r] * const_filter[offC + c * Rank + r];
    __syncthreads();
  }

  if (hBlockOff + h >= H) return;
  if (wBlockOff + w >= W) return;

  /****************************************************************************
   * Reduce over rank while scaling by kth filter value.
   ****************************************************************************/
  for (unsigned t = 0; t < T; ++t) {

    float tth_filter_pixel = 0.0f;

    for (unsigned r = 0; r < Rank; ++r)
      tth_filter_pixel
          += intermediate_c_acc[r] * const_filter[offT + t * Rank + r];

    Out[n * T * H * W + t * H * W + (h + hBlockOff) * W + w + wBlockOff]
        = tth_filter_pixel;
  }
}


float cp4_conv2d_forward_gpu(tensor_shape params,
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
  const unsigned offH = offC + (C * Rank);
  const unsigned offW = offH + (Y * Rank);
  ErrChk(cudaMemcpyToSymbol(
      const_filter, FT, sizeof(float) * (T * Rank), sizeof(float) * offT));
  ErrChk(cudaMemcpyToSymbol(
      const_filter, FC, sizeof(float) * (C * Rank), sizeof(float) * offC));
  ErrChk(cudaMemcpyToSymbol(
      const_filter, FY, sizeof(float) * (Y * Rank), sizeof(float) * offH));
  ErrChk(cudaMemcpyToSymbol(
      const_filter, FX, sizeof(float) * (X * Rank), sizeof(float) * offW));

  cudaDeviceProp prop;
  ErrChk(cudaGetDeviceProperties(&prop, 0));

  unsigned Bh   = 4;
  unsigned Bw   = 16;
  unsigned sW   = X - 1 + Bw;
  unsigned sH   = Y - 1 + Bh;
  size_t   smsz = sW * sH * sizeof(float);

  if (smsz > prop.sharedMemPerBlock) {
    cerr << "Shared Mem Too Big! " << smsz << " > " << prop.sharedMemPerBlock
         << endl;
  }

  const unsigned WgrdDim = (W / Bw) + ((W % Bw) != 0);
  const unsigned HgrdDim = (H / Bh) + ((H % Bh) != 0);
  const dim3     Gshp(WgrdDim, HgrdDim, N);
  const dim3     Bshp(Bw, Bh, 1);

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
          case  1: conv2d_cp4_kernel< 1, 1><<<Gshp, Bshp, smsz>>>(Out, In, N, C, H, W, pad, offT, offC, offH, offW, T, sW, sH); break;
          case  2: conv2d_cp4_kernel< 1, 2><<<Gshp, Bshp, smsz>>>(Out, In, N, C, H, W, pad, offT, offC, offH, offW, T, sW, sH); break;
          case  4: conv2d_cp4_kernel< 1, 4><<<Gshp, Bshp, smsz>>>(Out, In, N, C, H, W, pad, offT, offC, offH, offW, T, sW, sH); break;
          case  8: conv2d_cp4_kernel< 1, 8><<<Gshp, Bshp, smsz>>>(Out, In, N, C, H, W, pad, offT, offC, offH, offW, T, sW, sH); break;
          case 16: conv2d_cp4_kernel< 1,16><<<Gshp, Bshp, smsz>>>(Out, In, N, C, H, W, pad, offT, offC, offH, offW, T, sW, sH); break;
          default: cerr << "Rank not supported!" << endl;
        }
        break;
      case 3:
        switch (Rank) {
          case  1: conv2d_cp4_kernel< 3, 1><<<Gshp, Bshp, smsz>>>(Out, In, N, C, H, W, pad, offT, offC, offH, offW, T, sW, sH); break;
          case  2: conv2d_cp4_kernel< 3, 2><<<Gshp, Bshp, smsz>>>(Out, In, N, C, H, W, pad, offT, offC, offH, offW, T, sW, sH); break;
          case  4: conv2d_cp4_kernel< 3, 4><<<Gshp, Bshp, smsz>>>(Out, In, N, C, H, W, pad, offT, offC, offH, offW, T, sW, sH); break;
          case  8: conv2d_cp4_kernel< 3, 8><<<Gshp, Bshp, smsz>>>(Out, In, N, C, H, W, pad, offT, offC, offH, offW, T, sW, sH); break;
          case 16: conv2d_cp4_kernel< 3,16><<<Gshp, Bshp, smsz>>>(Out, In, N, C, H, W, pad, offT, offC, offH, offW, T, sW, sH); break;
          default: cerr << "Rank not supported!" << endl;
        }
        break;
      case 5:
        switch (Rank) {
          case  1: conv2d_cp4_kernel< 5, 1><<<Gshp, Bshp, smsz>>>(Out, In, N, C, H, W, pad, offT, offC, offH, offW, T, sW, sH); break;
          case  2: conv2d_cp4_kernel< 5, 2><<<Gshp, Bshp, smsz>>>(Out, In, N, C, H, W, pad, offT, offC, offH, offW, T, sW, sH); break;
          case  4: conv2d_cp4_kernel< 5, 4><<<Gshp, Bshp, smsz>>>(Out, In, N, C, H, W, pad, offT, offC, offH, offW, T, sW, sH); break;
          case  8: conv2d_cp4_kernel< 5, 8><<<Gshp, Bshp, smsz>>>(Out, In, N, C, H, W, pad, offT, offC, offH, offW, T, sW, sH); break;
          case 16: conv2d_cp4_kernel< 5,16><<<Gshp, Bshp, smsz>>>(Out, In, N, C, H, W, pad, offT, offC, offH, offW, T, sW, sH); break;
          default: cerr << "Rank not supported!" << endl;
        }
        break;
      case 7:
        switch (Rank) {
          case  1: conv2d_cp4_kernel< 7, 1><<<Gshp, Bshp, smsz>>>(Out, In, N, C, H, W, pad, offT, offC, offH, offW, T, sW, sH); break;
          case  2: conv2d_cp4_kernel< 7, 2><<<Gshp, Bshp, smsz>>>(Out, In, N, C, H, W, pad, offT, offC, offH, offW, T, sW, sH); break;
          case  4: conv2d_cp4_kernel< 7, 4><<<Gshp, Bshp, smsz>>>(Out, In, N, C, H, W, pad, offT, offC, offH, offW, T, sW, sH); break;
          case  8: conv2d_cp4_kernel< 7, 8><<<Gshp, Bshp, smsz>>>(Out, In, N, C, H, W, pad, offT, offC, offH, offW, T, sW, sH); break;
          case 16: conv2d_cp4_kernel< 7,16><<<Gshp, Bshp, smsz>>>(Out, In, N, C, H, W, pad, offT, offC, offH, offW, T, sW, sH); break;
          default: cerr << "Rank not supported!" << endl;
        }
        break;
      case 9:
        switch (Rank) {
          case  1: conv2d_cp4_kernel< 9, 1><<<Gshp, Bshp, smsz>>>(Out, In, N, C, H, W, pad, offT, offC, offH, offW, T, sW, sH); break;
          case  2: conv2d_cp4_kernel< 9, 2><<<Gshp, Bshp, smsz>>>(Out, In, N, C, H, W, pad, offT, offC, offH, offW, T, sW, sH); break;
          case  4: conv2d_cp4_kernel< 9, 4><<<Gshp, Bshp, smsz>>>(Out, In, N, C, H, W, pad, offT, offC, offH, offW, T, sW, sH); break;
          case  8: conv2d_cp4_kernel< 9, 8><<<Gshp, Bshp, smsz>>>(Out, In, N, C, H, W, pad, offT, offC, offH, offW, T, sW, sH); break;
          case 16: conv2d_cp4_kernel< 9,16><<<Gshp, Bshp, smsz>>>(Out, In, N, C, H, W, pad, offT, offC, offH, offW, T, sW, sH); break;
          default: cerr << "Rank not supported!" << endl;
        }
        break;
      case 11:
        switch (Rank) {
          case  1: conv2d_cp4_kernel<11, 1><<<Gshp, Bshp, smsz>>>(Out, In, N, C, H, W, pad, offT, offC, offH, offW, T, sW, sH); break;
          case  2: conv2d_cp4_kernel<11, 2><<<Gshp, Bshp, smsz>>>(Out, In, N, C, H, W, pad, offT, offC, offH, offW, T, sW, sH); break;
          case  4: conv2d_cp4_kernel<11, 4><<<Gshp, Bshp, smsz>>>(Out, In, N, C, H, W, pad, offT, offC, offH, offW, T, sW, sH); break;
          case  8: conv2d_cp4_kernel<11, 8><<<Gshp, Bshp, smsz>>>(Out, In, N, C, H, W, pad, offT, offC, offH, offW, T, sW, sH); break;
          case 16: conv2d_cp4_kernel<11,16><<<Gshp, Bshp, smsz>>>(Out, In, N, C, H, W, pad, offT, offC, offH, offW, T, sW, sH); break;
          default: cerr << "Rank not supported!" << endl;
        }
        break;
      case 13:
        switch (Rank) {
          case  1: conv2d_cp4_kernel<13, 1><<<Gshp, Bshp, smsz>>>(Out, In, N, C, H, W, pad, offT, offC, offH, offW, T, sW, sH); break;
          case  2: conv2d_cp4_kernel<13, 2><<<Gshp, Bshp, smsz>>>(Out, In, N, C, H, W, pad, offT, offC, offH, offW, T, sW, sH); break;
          case  4: conv2d_cp4_kernel<13, 4><<<Gshp, Bshp, smsz>>>(Out, In, N, C, H, W, pad, offT, offC, offH, offW, T, sW, sH); break;
          case  8: conv2d_cp4_kernel<13, 8><<<Gshp, Bshp, smsz>>>(Out, In, N, C, H, W, pad, offT, offC, offH, offW, T, sW, sH); break;
          case 16: conv2d_cp4_kernel<13,16><<<Gshp, Bshp, smsz>>>(Out, In, N, C, H, W, pad, offT, offC, offH, offW, T, sW, sH); break;
          default: cerr << "Rank not supported!" << endl;
        }
        break;
      case 15:
        switch (Rank) {
          case  1: conv2d_cp4_kernel<15, 1><<<Gshp, Bshp, smsz>>>(Out, In, N, C, H, W, pad, offT, offC, offH, offW, T, sW, sH); break;
          case  2: conv2d_cp4_kernel<15, 2><<<Gshp, Bshp, smsz>>>(Out, In, N, C, H, W, pad, offT, offC, offH, offW, T, sW, sH); break;
          case  4: conv2d_cp4_kernel<15, 4><<<Gshp, Bshp, smsz>>>(Out, In, N, C, H, W, pad, offT, offC, offH, offW, T, sW, sH); break;
          case  8: conv2d_cp4_kernel<15, 8><<<Gshp, Bshp, smsz>>>(Out, In, N, C, H, W, pad, offT, offC, offH, offW, T, sW, sH); break;
          case 16: conv2d_cp4_kernel<15,16><<<Gshp, Bshp, smsz>>>(Out, In, N, C, H, W, pad, offT, offC, offH, offW, T, sW, sH); break;
          default: cerr << "Rank not supported!" << endl;
        }
        break;
      case 17:
        switch (Rank) {
          case  1: conv2d_cp4_kernel<17, 1><<<Gshp, Bshp, smsz>>>(Out, In, N, C, H, W, pad, offT, offC, offH, offW, T, sW, sH); break;
          case  2: conv2d_cp4_kernel<17, 2><<<Gshp, Bshp, smsz>>>(Out, In, N, C, H, W, pad, offT, offC, offH, offW, T, sW, sH); break;
          case  4: conv2d_cp4_kernel<17, 4><<<Gshp, Bshp, smsz>>>(Out, In, N, C, H, W, pad, offT, offC, offH, offW, T, sW, sH); break;
          case  8: conv2d_cp4_kernel<17, 8><<<Gshp, Bshp, smsz>>>(Out, In, N, C, H, W, pad, offT, offC, offH, offW, T, sW, sH); break;
          case 16: conv2d_cp4_kernel<17,16><<<Gshp, Bshp, smsz>>>(Out, In, N, C, H, W, pad, offT, offC, offH, offW, T, sW, sH); break;
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
