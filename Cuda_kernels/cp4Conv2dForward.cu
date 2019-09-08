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


template <unsigned tile_sz>
__device__ __inline__ float
reduce_sum_tile_shfl(cg::thread_block_tile<tile_sz> g, float val) {
  // Each iteration halves the number of active threads
  // Each thread adds its partial sum[i] to sum[lane+i]
  for (int i = g.size() >> 1; i > 0; i >>= 1) { val += g.shfl_down(val, i); }

  return val; // note: only thread 0 will return full sum
}

/*******************************************************************************
 * 2 Dimensional Convolution Operation using an order-4 CP decomposition.
 * Also known as a Candecomp/Parafac Decomposition, a Canonical Polyadic
 * Decomposition, and a Tensor RANK Decomposition.
 *******************************************************************************/
template <unsigned CHANNEL_DIM, unsigned RANK>
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
                                  const unsigned Y,
                                  const unsigned X,
                                  const unsigned sW,
                                  const unsigned sH) {

  /* extern __shared__ float shared_mem[]; */

  // Threads
  const unsigned ct = threadIdx.x;
  const unsigned w  = threadIdx.y;
  const unsigned h  = threadIdx.z;

  // Block Dimensions
  const unsigned Bc = blockDim.x;
  const unsigned Bw = blockDim.y;
  const unsigned Bh = blockDim.z;

  // Offsets
  const unsigned wBlockOff = blockIdx.x * Bw;
  const unsigned hBlockOff = blockIdx.y * Bh;
  const unsigned n         = blockIdx.z;

  // Shared Memory
  /* float* work_mem = &shared_mem[(h * Bw * Bc) + (w * Bc)]; */
  /* work_mem[ct] = 0.0f; */

  float c_acc[RANK];
  for (unsigned r = 0; r < RANK; ++r) c_acc[r] = 0.0f;

  for (unsigned c = ct; c < C; c += Bc) {
    /* auto active = cg::coalesced_threads(); */
    // Shift the Global pointers to our Region Of interest
    const float* iPtr = Input + n * C * H * W + c * H * W;

    float pix_acc[RANK];
    for (unsigned r = 0; r < RANK; ++r) pix_acc[r] = 0.0f;

    for (unsigned y = 0; y < Y; ++y)
      for (unsigned x = 0; x < X; ++x) {
        if (y + h + hBlockOff >= pad && y + h + hBlockOff < H + pad
            && x + w + wBlockOff >= pad
            && x + w + wBlockOff < W + pad)
#pragma unroll
          for (unsigned r = 0; r < RANK; ++r)
            pix_acc[r] +=
                iPtr[(h + y + hBlockOff - pad) * W + (w + x + wBlockOff - pad)]
                * const_filter[offH + y * RANK + r]
                * const_filter[offW + x * RANK + r];
      }

    for (unsigned r = 0; r < RANK; ++r)
      c_acc[r] += pix_acc[r] * const_filter[offC + c * RANK + r];
  }

  /* if (hBlockOff + h >= H) return; */
  /* if (wBlockOff + w >= W) return; */

  /****************************************************************************
   * Reduce over RANK while scaling by kth filter value.
   ****************************************************************************/
  for (unsigned t = 0; t < T; ++t) {

    float out_acc = 0.0f;

#pragma unroll
    for (unsigned r = 0; r < RANK; ++r)
      out_acc += c_acc[r] * const_filter[offT + t * RANK + r];

    auto tile = cg::tiled_partition<CHANNEL_DIM>(cg::this_thread_block());
    out_acc   = reduce_sum_tile_shfl<CHANNEL_DIM>(tile, out_acc);

    cg::sync(tile);

    if (ct == 0)
      if (hBlockOff + h < H)
        if (wBlockOff + w < W)
          Out[n * T * H * W + t * H * W + (h + hBlockOff) * W + w + wBlockOff] =
              out_acc;
  }
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
/******************************************************************************
   Compute the next highest power of 2 for an unsigned integer
 *****************************************************************************/
unsigned next_highest_power_2(unsigned n) {
  if (n == 0) return 1;
  n--;
  n |= n >> 1;
  n |= n >> 2;
  n |= n >> 4;
  n |= n >> 8;
  n |= n >> 16;
  n++;
  return n;
}

/******************************************************************************
   Compute the next lowest power of 2.
 *****************************************************************************/
unsigned next_lowest_power_2(unsigned n) {
  n |= (n >> 1);
  n |= (n >> 2);
  n |= (n >> 4);
  n |= (n >> 8);
  n |= (n >> 16);
  return n - (n >> 1);
}

/******************************************************************************
   Compute the next lowest power of 2.
 *****************************************************************************/
unsigned log_2(unsigned n, unsigned step = 1) {
  unsigned int r = 0; // r will be lg(v)

  while (n >>= step) r++;
  return r;
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

  unsigned Bw   = W < 16 ? 1 : W < 32 ? 4 : W > 128 ? 16 : 8;
  unsigned Bh   = H < 32 ? 1 : H < 32 ? 2 : H > 128 ? 8 : 4;
  unsigned Bc   = C < 32 ? 1 : C > 128 ? 32 : 8;
  unsigned sW   = X - 1 + Bw;
  unsigned sH   = Y - 1 + Bh;
  size_t   smsz = 0; // * ((Bc * Bw * Bh)) * sizeof(float);

  if (smsz > prop.sharedMemPerBlock) {
    cerr << "Shared Mem Too Big! " << smsz << " > " << prop.sharedMemPerBlock
         << endl;
  }

  const unsigned WgrdDim = (W / Bw) + ((W % Bw) != 0);
  const unsigned HgrdDim = (H / Bh) + ((H % Bh) != 0);
  const dim3     Gshp(WgrdDim, HgrdDim, N);
  const dim3     Bshp(Bc, Bw, Bh);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float us = 0.0f;

  for (unsigned i = 0; i < PROFCOUNT; ++i) {
    ErrChk(cudaDeviceSynchronize());
    cudaEventRecord(start);
    // clang-format off
    switch (Bc) {
      case 1:
        switch (Rank) {
          case  1: conv2d_cp4_kernel<1, 1><<<Gshp, Bshp, smsz>>>(Out, In, N, C, H, W, pad, offT, offC, offH, offW, T, Y, X, sW, sH); break;
          case  2: conv2d_cp4_kernel<1, 2><<<Gshp, Bshp, smsz>>>(Out, In, N, C, H, W, pad, offT, offC, offH, offW, T, Y, X, sW, sH); break;
          case  4: conv2d_cp4_kernel<1, 4><<<Gshp, Bshp, smsz>>>(Out, In, N, C, H, W, pad, offT, offC, offH, offW, T, Y, X, sW, sH); break;
          case  8: conv2d_cp4_kernel<1, 8><<<Gshp, Bshp, smsz>>>(Out, In, N, C, H, W, pad, offT, offC, offH, offW, T, Y, X, sW, sH); break;
          case 16: conv2d_cp4_kernel<1,16><<<Gshp, Bshp, smsz>>>(Out, In, N, C, H, W, pad, offT, offC, offH, offW, T, Y, X, sW, sH); break;
          default: cerr << "Rank not supported!" << endl;
        } break;
      case 2:
        switch (Rank) {
          case  1: conv2d_cp4_kernel<2, 1><<<Gshp, Bshp, smsz>>>(Out, In, N, C, H, W, pad, offT, offC, offH, offW, T, Y, X, sW, sH); break;
          case  2: conv2d_cp4_kernel<2, 2><<<Gshp, Bshp, smsz>>>(Out, In, N, C, H, W, pad, offT, offC, offH, offW, T, Y, X, sW, sH); break;
          case  4: conv2d_cp4_kernel<2, 4><<<Gshp, Bshp, smsz>>>(Out, In, N, C, H, W, pad, offT, offC, offH, offW, T, Y, X, sW, sH); break;
          case  8: conv2d_cp4_kernel<2, 8><<<Gshp, Bshp, smsz>>>(Out, In, N, C, H, W, pad, offT, offC, offH, offW, T, Y, X, sW, sH); break;
          case 16: conv2d_cp4_kernel<2,16><<<Gshp, Bshp, smsz>>>(Out, In, N, C, H, W, pad, offT, offC, offH, offW, T, Y, X, sW, sH); break;
          default: cerr << "Rank not supported!" << endl;
        } break;
      case 4:
        switch (Rank) {
          case  1: conv2d_cp4_kernel<4, 1><<<Gshp, Bshp, smsz>>>(Out, In, N, C, H, W, pad, offT, offC, offH, offW, T, Y, X, sW, sH); break;
          case  2: conv2d_cp4_kernel<4, 2><<<Gshp, Bshp, smsz>>>(Out, In, N, C, H, W, pad, offT, offC, offH, offW, T, Y, X, sW, sH); break;
          case  4: conv2d_cp4_kernel<4, 4><<<Gshp, Bshp, smsz>>>(Out, In, N, C, H, W, pad, offT, offC, offH, offW, T, Y, X, sW, sH); break;
          case  8: conv2d_cp4_kernel<4, 8><<<Gshp, Bshp, smsz>>>(Out, In, N, C, H, W, pad, offT, offC, offH, offW, T, Y, X, sW, sH); break;
          case 16: conv2d_cp4_kernel<4,16><<<Gshp, Bshp, smsz>>>(Out, In, N, C, H, W, pad, offT, offC, offH, offW, T, Y, X, sW, sH); break;
          default: cerr << "Rank not supported!" << endl;
        } break;
      case 8:
        switch (Rank) {
          case  1: conv2d_cp4_kernel<8, 1><<<Gshp, Bshp, smsz>>>(Out, In, N, C, H, W, pad, offT, offC, offH, offW, T, Y, X, sW, sH); break;
          case  2: conv2d_cp4_kernel<8, 2><<<Gshp, Bshp, smsz>>>(Out, In, N, C, H, W, pad, offT, offC, offH, offW, T, Y, X, sW, sH); break;
          case  4: conv2d_cp4_kernel<8, 4><<<Gshp, Bshp, smsz>>>(Out, In, N, C, H, W, pad, offT, offC, offH, offW, T, Y, X, sW, sH); break;
          case  8: conv2d_cp4_kernel<8, 8><<<Gshp, Bshp, smsz>>>(Out, In, N, C, H, W, pad, offT, offC, offH, offW, T, Y, X, sW, sH); break;
          case 16: conv2d_cp4_kernel<8,16><<<Gshp, Bshp, smsz>>>(Out, In, N, C, H, W, pad, offT, offC, offH, offW, T, Y, X, sW, sH); break;
          default: cerr << "Rank not supported!" << endl;
        } break;
      case 16:
        switch (Rank) {
          case  1: conv2d_cp4_kernel<16, 1><<<Gshp, Bshp, smsz>>>(Out, In, N, C, H, W, pad, offT, offC, offH, offW, T, Y, X, sW, sH); break;
          case  2: conv2d_cp4_kernel<16, 2><<<Gshp, Bshp, smsz>>>(Out, In, N, C, H, W, pad, offT, offC, offH, offW, T, Y, X, sW, sH); break;
          case  4: conv2d_cp4_kernel<16, 4><<<Gshp, Bshp, smsz>>>(Out, In, N, C, H, W, pad, offT, offC, offH, offW, T, Y, X, sW, sH); break;
          case  8: conv2d_cp4_kernel<16, 8><<<Gshp, Bshp, smsz>>>(Out, In, N, C, H, W, pad, offT, offC, offH, offW, T, Y, X, sW, sH); break;
          case 16: conv2d_cp4_kernel<16,16><<<Gshp, Bshp, smsz>>>(Out, In, N, C, H, W, pad, offT, offC, offH, offW, T, Y, X, sW, sH); break;
          default: cerr << "Rank not supported!" << endl;
        } break;
      case 32:
        switch (Rank) {
          case  1: conv2d_cp4_kernel<32, 1><<<Gshp, Bshp, smsz>>>(Out, In, N, C, H, W, pad, offT, offC, offH, offW, T, Y, X, sW, sH); break;
          case  2: conv2d_cp4_kernel<32, 2><<<Gshp, Bshp, smsz>>>(Out, In, N, C, H, W, pad, offT, offC, offH, offW, T, Y, X, sW, sH); break;
          case  4: conv2d_cp4_kernel<32, 4><<<Gshp, Bshp, smsz>>>(Out, In, N, C, H, W, pad, offT, offC, offH, offW, T, Y, X, sW, sH); break;
          case  8: conv2d_cp4_kernel<32, 8><<<Gshp, Bshp, smsz>>>(Out, In, N, C, H, W, pad, offT, offC, offH, offW, T, Y, X, sW, sH); break;
          case 16: conv2d_cp4_kernel<32,16><<<Gshp, Bshp, smsz>>>(Out, In, N, C, H, W, pad, offT, offC, offH, offW, T, Y, X, sW, sH); break;
          default: cerr << "Rank not supported!" << endl;
        } break;

    }
    // clang-format on

    ErrChk(cudaPeekAtLastError());
    ErrChk(cudaDeviceSynchronize());
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    us += milliseconds * 1e3;
  }
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  us = us / PROFCOUNT;

  return us;
}
