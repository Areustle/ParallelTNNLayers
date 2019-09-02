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


template <int tile_sz>
__device__ float
reduce_sum_tile_shfl(cg::thread_block_tile<tile_sz> g, float val) {
  // Each iteration halves the number of active threads
  // Each thread adds its partial sum[i] to sum[lane+i]
  for (int i = g.size() / 2; i > 0; i /= 2) { val += g.shfl_down(val, i); }

  return val; // note: only thread 0 will return full sum
}


/*******************************************************************************
 * 2 Dimensional Convolution Operation using an order-4 CP decomposition.
 * Also known as a Candecomp/Parafac Decomposition, a Canonical Polyadic
 * Decomposition, and a Tensor Rank Decomposition.
 *******************************************************************************/
template <unsigned CHANNEL_DIM, unsigned RANK_DIM>
__global__ void conv2d_cp4_kernel(float* __restrict__ Out,
                                  const float* __restrict__ Input,
                                  const unsigned H,
                                  const unsigned W,
                                  const unsigned pad,
                                  const unsigned offT,
                                  const unsigned offC,
                                  const unsigned offY,
                                  const unsigned offX,
                                  const unsigned T,
                                  const unsigned C,
                                  const unsigned Y,
                                  const unsigned X,
                                  const unsigned Rank,
                                  const unsigned Bh,
                                  const unsigned Bw) {

  extern __shared__ float shared_mem[];

  const float* FT = const_filter + offT;
  const float* FC = const_filter + offC;
  const float* FY = const_filter + offY;
  const float* FX = const_filter + offX;

  auto ChannelWarp = cg::tiled_partition<CHANNEL_DIM>(cg::this_thread_block());

  const unsigned n  = blockIdx.z;
  unsigned       tc = (threadIdx.x) % CHANNEL_DIM;
  unsigned       r  = (threadIdx.x / CHANNEL_DIM) % RANK_DIM;
  unsigned       w  = (threadIdx.x / CHANNEL_DIM / RANK_DIM) % Bw;
  unsigned       h  = (threadIdx.x / CHANNEL_DIM / RANK_DIM / Bw);

  const unsigned wBlockOff = blockIdx.x * Bw;
  const unsigned hBlockOff = blockIdx.y * Bh;
  const unsigned sH        = Y - 1 + Bh;
  const unsigned sW        = X - 1 + Bw;

  if (h+hBlockOff >= H) return;
  if (w+wBlockOff >= W) return;
  if (r >= Rank) return;

  float local_acc = 0.0f;

  // Sub Block of "tc" threads for shared memory load.
  unsigned SubBlockDim   = blockDim.x / CHANNEL_DIM;
  unsigned SubBlockDim_H = Bh;
  unsigned SubBlockDim_W = SubBlockDim / SubBlockDim_H;

  unsigned stc = threadIdx.x / SubBlockDim;
  unsigned stt = threadIdx.x % SubBlockDim;
  unsigned sth = stt / SubBlockDim_W;
  unsigned stw = stt % SubBlockDim_W;

  for (unsigned rc = tc; rc < C; rc += CHANNEL_DIM) {

    auto     active = cg::coalesced_threads();
    unsigned c      = (rc / CHANNEL_DIM) + stc;

    const float* iPtr      = Input + n * C * H * W + c * H * W;
    float        local_pix = 0.0f;

    for (unsigned y = 0; y < Y; ++y) {
      for (unsigned x = 0; x < X; ++x) {
        if (h + hBlockOff + y >= pad       //
            && h + hBlockOff + y < H + pad //
            && w + wBlockOff + x >= pad    //
            && w + wBlockOff + x < W + pad)
          local_pix +=
              iPtr[(h + hBlockOff + y - pad) * H + (w + wBlockOff + x - pad)]
              * FX[x * Rank + r] * FY[y * Rank + r];
      }
    }

    /* for (unsigned y = sth; y < sH; y += SubBlockDim_H) */
    /*   for (unsigned x = stw; x < sW; x += SubBlockDim_W) */
    /*     shared_mem[stc * sH * sW + y * sW + x] = */
    /*         (y + sth + hBlockOff >= pad       // */
    /*          && y + sth + hBlockOff < H + pad // */
    /*          && x + stw + wBlockOff >= pad    // */
    /*          && x + stw + wBlockOff < W + pad) */
    /*             ? iPtr[(y + sth + hBlockOff - pad) * W */
    /*                    + (x + stw + wBlockOff - pad)] */
    /*             : (0.0f); // Pad with Zeros if outside the bounds */

    /* active.sync(); */

    /* for (unsigned y = 0; y < sH; ++y) { */
    /*   for (unsigned x = 0; x < sW; ++x) { */
    /*     local_pix += shared_mem[tc * sH * sW + y * sW + x] * FX[x * Rank + r] */
    /*                  * FY[y * Rank + r]; */
    /*   } */
    /* } */
    local_acc += local_pix * FC[tc * Rank + r];
  }

  float* work_mem = shared_mem + h * Bw * Rank + w * Rank;

  local_acc = reduce_sum_tile_shfl<CHANNEL_DIM>(ChannelWarp, local_acc);

  // Save intermediate rank vector to shared memory
  if (ChannelWarp.thread_rank() == 0) work_mem[r] = local_acc;
  __syncthreads();

  // Swap which threads are in the warp.
  tc = (threadIdx.x / RANK_DIM) % CHANNEL_DIM;
  r  = threadIdx.x % RANK_DIM;

  auto RankWarp = cg::tiled_partition<RANK_DIM>(cg::this_thread_block());

  // scatter rank vector to all threads.
  local_acc = work_mem[r];
  /* float rvec[RANK_DIM]; */
  /* for (int i = 0; i < RANK_DIM; ++i) rvec[i] = work_mem[i]; */

  // parallel scale all output channels and sum over rank
  for (unsigned t = tc; t < T; t += CHANNEL_DIM) {

    float output_acc = local_acc * FT[t * Rank + r];
    /* float output_acc = 0.0; */
    /* for (int i = 0; i < RANK_DIM; ++i) output_acc += rvec[i] * FT[t * Rank +
     * r]; */
    output_acc = reduce_sum_tile_shfl<RANK_DIM>(RankWarp, output_acc);

    // Output result to global memory
    if (r == 0) /* if (RankWarp.thread_rank() == 0) */
      Out[n * T * H * W + t * H * W + (h + hBlockOff) * W + w + wBlockOff] =
          output_acc;
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
unsigned log_2(unsigned n) {
  unsigned int r = 0; // r will be lg(v)

  while (n >>= 1) r++;
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

  unsigned ChannelPow2 = min(32, next_highest_power_2(max(C,T)));
  unsigned RankPow2    = min(32, next_lowest_power_2(Rank));
  unsigned RankLog2    = min(32, log_2(Rank));
  /* cout << ChannelPow2 << " " << RankPow2 << endl; */

  unsigned NumThreads    = 256;
  unsigned BlockSize     = min(NumThreads, Rank * ChannelPow2);
  unsigned TileC         = BlockSize / Rank;
  BlockSize              = TileC * Rank;
  const unsigned spatial = intSqrt(NumThreads / BlockSize);
  const unsigned Bh      = min(spatial, H);
  const unsigned Bw      = min(spatial, W);
  const unsigned sH      = Y - 1 + Bh;
  const unsigned sW      = X - 1 + Bw;
  BlockSize *= Bh * Bw;
  /* cout << BlockSize << " " << TileC << " " << Rank << " " << Bh << " " << Bw */
       /* << endl; */

  size_t smsz = max(Rank * Bh * Bw, sH * sW * TileC) * sizeof(float);

  if (smsz > prop.sharedMemPerBlock) {
    cerr << "Shared Mem Too Big! " << smsz << " > " << prop.sharedMemPerBlock
         << endl;
  }

  const unsigned WgrdDim = (W / Bw) + ((W % Bw) != 0);
  const unsigned HgrdDim = (H / Bh) + ((H % Bh) != 0);
  /* cout << WgrdDim << " " << HgrdDim << endl; */

  const dim3 Gshp(WgrdDim, HgrdDim, N);
  const dim3 Bshp(BlockSize);
  /* const dim3 Bshp(TileC, Bh * Bw, Rank); */

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float us = 0.0f;

  for (unsigned i = 0; i < PROFCOUNT; ++i) {
    ErrChk(cudaDeviceSynchronize());
    cudaEventRecord(start);

    // clang-format off
    switch(ChannelPow2) {
      case 32:
        switch(RankPow2) {
          case 32: conv2d_cp4_kernel<32,32><<<Gshp, Bshp, smsz>>>(Out, In, H, W, pad, offT, offC, offY, offX, T, C, Y, X, Rank, Bh, Bw); break;
          case 16: conv2d_cp4_kernel<32,16><<<Gshp, Bshp, smsz>>>(Out, In, H, W, pad, offT, offC, offY, offX, T, C, Y, X, Rank, Bh, Bw); break;
          case 8: conv2d_cp4_kernel<32,8><<<Gshp, Bshp, smsz>>>(Out, In, H, W, pad, offT, offC, offY, offX, T, C, Y, X, Rank, Bh, Bw); break;
          case 4: conv2d_cp4_kernel<32,4><<<Gshp, Bshp, smsz>>>(Out, In, H, W, pad, offT, offC, offY, offX, T, C, Y, X, Rank, Bh, Bw); break;
          case 2: conv2d_cp4_kernel<32,2><<<Gshp, Bshp, smsz>>>(Out, In, H, W, pad, offT, offC, offY, offX, T, C, Y, X, Rank, Bh, Bw); break;
          case 1: conv2d_cp4_kernel<32,1><<<Gshp, Bshp, smsz>>>(Out, In, H, W, pad, offT, offC, offY, offX, T, C, Y, X, Rank, Bh, Bw); break;
        }
               break;
      case 16:
        switch(RankPow2) {
          case 32: conv2d_cp4_kernel<16,32><<<Gshp, Bshp, smsz>>>(Out, In, H, W, pad, offT, offC, offY, offX, T, C, Y, X, Rank, Bh, Bw); break;
          case 16: conv2d_cp4_kernel<16,16><<<Gshp, Bshp, smsz>>>(Out, In, H, W, pad, offT, offC, offY, offX, T, C, Y, X, Rank, Bh, Bw); break;
          case 8: conv2d_cp4_kernel<16,8><<<Gshp, Bshp, smsz>>>(Out, In, H, W, pad, offT, offC, offY, offX, T, C, Y, X, Rank, Bh, Bw); break;
          case 4: conv2d_cp4_kernel<16,4><<<Gshp, Bshp, smsz>>>(Out, In, H, W, pad, offT, offC, offY, offX, T, C, Y, X, Rank, Bh, Bw); break;
          case 2: conv2d_cp4_kernel<16,2><<<Gshp, Bshp, smsz>>>(Out, In, H, W, pad, offT, offC, offY, offX, T, C, Y, X, Rank, Bh, Bw); break;
          case 1: conv2d_cp4_kernel<16,1><<<Gshp, Bshp, smsz>>>(Out, In, H, W, pad, offT, offC, offY, offX, T, C, Y, X, Rank, Bh, Bw); break;
        }
               break;
      case 8:
        switch(RankPow2) {
          case 32: conv2d_cp4_kernel<8,32><<<Gshp, Bshp, smsz>>>(Out, In, H, W, pad, offT, offC, offY, offX, T, C, Y, X, Rank, Bh, Bw); break;
          case 16: conv2d_cp4_kernel<8,16><<<Gshp, Bshp, smsz>>>(Out, In, H, W, pad, offT, offC, offY, offX, T, C, Y, X, Rank, Bh, Bw); break;
          case 8: conv2d_cp4_kernel<8,8><<<Gshp, Bshp, smsz>>>(Out, In, H, W, pad, offT, offC, offY, offX, T, C, Y, X, Rank, Bh, Bw); break;
          case 4: conv2d_cp4_kernel<8,4><<<Gshp, Bshp, smsz>>>(Out, In, H, W, pad, offT, offC, offY, offX, T, C, Y, X, Rank, Bh, Bw); break;
          case 2: conv2d_cp4_kernel<8,2><<<Gshp, Bshp, smsz>>>(Out, In, H, W, pad, offT, offC, offY, offX, T, C, Y, X, Rank, Bh, Bw); break;
          case 1: conv2d_cp4_kernel<8,1><<<Gshp, Bshp, smsz>>>(Out, In, H, W, pad, offT, offC, offY, offX, T, C, Y, X, Rank, Bh, Bw); break;
        }
               break;
      case 4:
        switch(RankPow2) {
          case 32: conv2d_cp4_kernel<4,32><<<Gshp, Bshp, smsz>>>(Out, In, H, W, pad, offT, offC, offY, offX, T, C, Y, X, Rank, Bh, Bw); break;
          case 16: conv2d_cp4_kernel<4,16><<<Gshp, Bshp, smsz>>>(Out, In, H, W, pad, offT, offC, offY, offX, T, C, Y, X, Rank, Bh, Bw); break;
          case 8: conv2d_cp4_kernel<4,8><<<Gshp, Bshp, smsz>>>(Out, In, H, W, pad, offT, offC, offY, offX, T, C, Y, X, Rank, Bh, Bw); break;
          case 4: conv2d_cp4_kernel<4,4><<<Gshp, Bshp, smsz>>>(Out, In, H, W, pad, offT, offC, offY, offX, T, C, Y, X, Rank, Bh, Bw); break;
          case 2: conv2d_cp4_kernel<4,2><<<Gshp, Bshp, smsz>>>(Out, In, H, W, pad, offT, offC, offY, offX, T, C, Y, X, Rank, Bh, Bw); break;
          case 1: conv2d_cp4_kernel<4,1><<<Gshp, Bshp, smsz>>>(Out, In, H, W, pad, offT, offC, offY, offX, T, C, Y, X, Rank, Bh, Bw); break;
        }
               break;
      case 2:
        switch(RankPow2) {
          case 32: conv2d_cp4_kernel<2,32><<<Gshp, Bshp, smsz>>>(Out, In, H, W, pad, offT, offC, offY, offX, T, C, Y, X, Rank, Bh, Bw); break;
          case 16: conv2d_cp4_kernel<2,16><<<Gshp, Bshp, smsz>>>(Out, In, H, W, pad, offT, offC, offY, offX, T, C, Y, X, Rank, Bh, Bw); break;
          case 8: conv2d_cp4_kernel<2,8><<<Gshp, Bshp, smsz>>>(Out, In, H, W, pad, offT, offC, offY, offX, T, C, Y, X, Rank, Bh, Bw); break;
          case 4: conv2d_cp4_kernel<2,4><<<Gshp, Bshp, smsz>>>(Out, In, H, W, pad, offT, offC, offY, offX, T, C, Y, X, Rank, Bh, Bw); break;
          case 2: conv2d_cp4_kernel<2,2><<<Gshp, Bshp, smsz>>>(Out, In, H, W, pad, offT, offC, offY, offX, T, C, Y, X, Rank, Bh, Bw); break;
          case 1: conv2d_cp4_kernel<2,1><<<Gshp, Bshp, smsz>>>(Out, In, H, W, pad, offT, offC, offY, offX, T, C, Y, X, Rank, Bh, Bw); break;
        }
               break;
      case 1:
        switch(RankPow2) {
          case 32: conv2d_cp4_kernel<1,32><<<Gshp, Bshp, smsz>>>(Out, In, H, W, pad, offT, offC, offY, offX, T, C, Y, X, Rank, Bh, Bw); break;
          case 16: conv2d_cp4_kernel<1,16><<<Gshp, Bshp, smsz>>>(Out, In, H, W, pad, offT, offC, offY, offX, T, C, Y, X, Rank, Bh, Bw); break;
          case 8: conv2d_cp4_kernel<1,8><<<Gshp, Bshp, smsz>>>(Out, In, H, W, pad, offT, offC, offY, offX, T, C, Y, X, Rank, Bh, Bw); break;
          case 4: conv2d_cp4_kernel<1,4><<<Gshp, Bshp, smsz>>>(Out, In, H, W, pad, offT, offC, offY, offX, T, C, Y, X, Rank, Bh, Bw); break;
          case 2: conv2d_cp4_kernel<1,2><<<Gshp, Bshp, smsz>>>(Out, In, H, W, pad, offT, offC, offY, offX, T, C, Y, X, Rank, Bh, Bw); break;
          case 1: conv2d_cp4_kernel<1,1><<<Gshp, Bshp, smsz>>>(Out, In, H, W, pad, offT, offC, offY, offX, T, C, Y, X, Rank, Bh, Bw); break;
        }
               break;
    }



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
