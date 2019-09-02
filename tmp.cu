#include "cp4Conv2dForward.cuh"
#include <cooperative_groups.h>
#include <iostream>
#include <stdlib.h>

using namespace std;
namespace cg = cooperative_groups;

// Simple cuda error checking macro
#define ErrChk(ans)                                                            \
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
   Hard coded limit to size of decomposed filter of 16384 floats = 128 KB
 ******************************************************************************/
__constant__ float const_filter[1 << 14];


/*******************************************************************************
 * 2 Dimensional Convolution Operation using an order-4 CP decomposition.
 * Also known as a Candecomp/Parafac Decomposition, a Canonical Polyadic
 * Decomposition, and a Tensor Rank Decomposition.
 *******************************************************************************/
template <unsigned FilterDim>
__global__ void    conv2d_cp4_kernel(float* __restrict__ Out,
                                  const float* __restrict__ Input,
                                  const unsigned pad,
                                  const unsigned offT,
                                  const unsigned offC,
                                  const unsigned offY,
                                  const unsigned offX,
                                  const unsigned T,
                                  const unsigned C,
                                  const unsigned Y,
                                  const unsigned X,
                                  const unsigned Rank) {

  extern __shared__ float shared_mem[];

  const float* FT = const_filter + offT;
  const float* FC = const_filter + offC;
  const float* FY = const_filter + offY;
  const float* FX = const_filter + offX;

  const unsigned w     = blockIdx.x;
  const unsigned h     = blockIdx.y;
  const unsigned n     = blockIdx.z;
  const unsigned W     = gridDim.x;
  const unsigned H     = gridDim.y;
  const unsigned TileC = blockDim.x / Rank;
  const unsigned r     = threadIdx.x / TileC;
  const unsigned tc    = threadIdx.x % TileC;
  const unsigned sH    = Y;
  const unsigned sW    = X;

  if (r >= Rank) return;
  if (tc >= TileC) return;

  float local_acc = 0.0f;

  for (unsigned c = tc; c < C; c += TileC) {
    auto active = cg::coalesced_threads();

    const float* iPtr      = Input + n * C * H * W + c * H * W;
    float        local_pix = 0.0f;
    // Cooperatively load all input segment into our shared memory and pad it.

    if (r == 0) {
#pragma unroll
      for (unsigned y = 0; y < FilterDim; ++y)
#pragma unroll
        for (unsigned x = 0; x < FilterDim; ++x)
          shared_mem[tc * sH * sW + y * sW + x] =
              (y + h >= pad       //
               && y + h < H + pad //
               && x + w >= pad    //
               && x + w < W + pad)
                  ? iPtr[(y + h - pad) * W + (x + w - pad)]
                  : (0.0f); // Pad with Zeros if outside the bounds
    }

    active.sync();

#pragma unroll
    for (unsigned y = 0; y < FilterDim; ++y) {
#pragma unroll
      for (unsigned x = 0; x < FilterDim; ++x) {
        if (h + y >= pad && h + y < H + pad && //
            w + x >= pad
            && w + x < W + pad)
          local_pix += shared_mem[tc * sH * sW + y * sW + x] * FX[x * Rank + r]
                       * FY[y * Rank + r];
      }
    }
    local_acc += local_pix * FC[c * Rank + r];
  }

  // Save each intermediate channel, rank value to shared memory
  __syncthreads();
  shared_mem[r * TileC + tc] = local_acc;
  __syncthreads();

  // reduce channels into single rank vector // gather
  local_acc = 0.0f;
  for (unsigned cc = 0; cc < TileC; cc++)
    local_acc += shared_mem[r * TileC + cc];

  // Save intermediate rank vector to shared memory
  __syncthreads();
  if (tc == 0) shared_mem[r] = local_acc;
  __syncthreads();

  // scatter rank vector to all threads.
  local_acc = shared_mem[r];

  // parallel scale all output channels and sum over rank
  for (unsigned t = tc; t < T; t += TileC) {
    float output_acc = 0.0f;

    __syncthreads();
    shared_mem[r * TileC + tc] = local_acc * FT[t * Rank + r];
    __syncthreads();

    for (unsigned rr = 0; rr < Rank; rr++)
      output_acc += shared_mem[rr * TileC + tc];

    // Output result to global memory
    if (r == 0) Out[n * T * H * W + t * H * W + h * W + w] = output_acc;
  }
}

/******************************************************************************
   Compute the next highest power of 2 for an unsigned integer
 *****************************************************************************/
unsigned next_power_of_2(unsigned v) {
  v--;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  v++;
  return v;
}

/******************************************************************************
   Compute the Integer square root of an unsigned integer.
 *****************************************************************************/
unsigned intSqrt(unsigned const n) {
  if (n < 2) return n;

  // Recursive call:
  unsigned p = intSqrt(n >> 2) << 1;
  unsigned q = p + 1;
  if (q * q > n) return p;
  return q;
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

  cudaDeviceProp prop;
  ErrChk(cudaGetDeviceProperties(&prop, 0));

  unsigned BlockSize = min(256, next_power_of_2(Rank * C));
  unsigned TileC     = BlockSize / Rank;
  BlockSize          = TileC * Rank;
  unsigned spatial   = intSqrt(256 / BlockSize);
  unsigned Bh        = spatial;
  unsigned Bw        = spatial;
  unsigned sH        = Y - 1 + Bh;
  unsigned sW        = X - 1 + Bw;
  BlockSize *= Bh * Bw;

  size_t smsz = max(BlockSize, (TileC * sH * sW)) * sizeof(float);

  if (smsz > prop.sharedMemPerBlock) {
    cerr << "Shared Mem Too Big! " << smsz << " > " << prop.sharedMemPerBlock
         << endl;
  }

  const dim3 Gshp(W, H, N);
  const dim3 Bshp(BlockSize);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float us = 0.0f;

  for (unsigned i = 0; i < PROFCOUNT; ++i) {
    ErrChk(cudaDeviceSynchronize());
    cudaEventRecord(start);
    // clang-format off
    switch (X){
      case 19 : conv2d_cp4_kernel<19><<<Gshp, Bshp, smsz>>>(Out, In, pad, offT, offC, offY, offX, T, C, Y, X, Rank);
                 break;
      case 17 : conv2d_cp4_kernel<17><<<Gshp, Bshp, smsz>>>(Out, In, pad, offT, offC, offY, offX, T, C, Y, X, Rank);
                 break;
      case 15 : conv2d_cp4_kernel<15><<<Gshp, Bshp, smsz>>>(Out, In, pad, offT, offC, offY, offX, T, C, Y, X, Rank);
                 break;
      case 13 : conv2d_cp4_kernel<13><<<Gshp, Bshp, smsz>>>(Out, In, pad, offT, offC, offY, offX, T, C, Y, X, Rank);
                break;
      case 11 : conv2d_cp4_kernel<11><<<Gshp, Bshp, smsz>>>(Out, In, pad, offT, offC, offY, offX, T, C, Y, X, Rank);
                break;
      case 9 : conv2d_cp4_kernel<9><<<Gshp, Bshp, smsz>>>(Out, In, pad, offT, offC, offY, offX, T, C, Y, X, Rank);
                break;
      case 7 : conv2d_cp4_kernel<7><<<Gshp, Bshp, smsz>>>(Out, In, pad, offT, offC, offY, offX, T, C, Y, X, Rank);
               break;
      case 5 : conv2d_cp4_kernel<5><<<Gshp, Bshp, smsz>>>(Out, In, pad, offT, offC, offY, offX, T, C, Y, X, Rank);
               break;
      case 3 : conv2d_cp4_kernel<3><<<Gshp, Bshp, smsz>>>(Out, In, pad, offT, offC, offY, offX, T, C, Y, X, Rank);
               break;
      case 1 : conv2d_cp4_kernel<1><<<Gshp, Bshp, smsz>>>(Out, In, pad, offT, offC, offY, offX, T, C, Y, X, Rank);
               break;
      default :
               cerr << "Block Size Not Supported! " << BlockSize << endl;
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

/*******************************************************************************
 * 2 Dimensional Convolution Operation using an order-4 CP decomposition.
 * Also known as a Candecomp/Parafac Decomposition, a Canonical Polyadic
 * Decomposition, and a Tensor Rank Decomposition.
 *******************************************************************************/
template <unsigned FilterDim, unsigned Rank>
__global__ void old_conv2d_cp4_kernel(float* __restrict__ Out,
                                      const float* __restrict__ Input,
                                      const unsigned N,
                                      const unsigned C,
                                      const unsigned H,
                                      const unsigned W,
                                      const unsigned pad,
                                      const unsigned offT,
                                      const unsigned offC,
                                      const unsigned offY,
                                      const unsigned offX,
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
        shared_mem[j * sW + i] =
            (j + hBlockOff >= pad       //
             && j + hBlockOff < H + pad //
             && i + wBlockOff >= pad    //
             && i + wBlockOff < W + pad)
                ? iPtr[(j + hBlockOff - pad) * W + (i + wBlockOff - pad)]
                : (0.0f); // Pad with Zeros if outside the bounds

    __syncthreads();

    float intermediate_pix_acc[Rank];
    for (unsigned r = 0; r < Rank; ++r) intermediate_pix_acc[r] = 0.0f;


    for (unsigned y = 0; y < FilterDim; ++y)
      for (unsigned x = 0; x < FilterDim; ++x) {
#pragma unroll
        for (unsigned r = 0; r < Rank; ++r)
          intermediate_pix_acc[r] += shared_mem[(h + y) * sW + (w + x)]
                                     * const_filter[offY + y * Rank + r]
                                     * const_filter[offX + x * Rank + r];
      }
    for (unsigned r = 0; r < Rank; ++r)
      intermediate_c_acc[r] +=
          intermediate_pix_acc[r] * const_filter[offC + c * Rank + r];
    __syncthreads();
  }

  // Handle block / input size mismatch. This occurs here and not earlier
  // So that these threads can still participate in the cooperative shared
  // Memory load.
  if (hBlockOff + h >= H) return;
  if (wBlockOff + w >= W) return;

  /****************************************************************************
   * Reduce over rank while scaling by kth filter value.
   ****************************************************************************/
  for (unsigned t = 0; t < T; ++t) {

    float tth_filter_pixel = 0.0f;

    for (unsigned r = 0; r < Rank; ++r)
      tth_filter_pixel +=
          intermediate_c_acc[r] * const_filter[offT + t * Rank + r];

    Out[n * T * H * W + t * H * W + (h + hBlockOff) * W + w + wBlockOff] =
        tth_filter_pixel;
  }
}
