#include "cp4Conv2dForward.cuh"
#include <cooperative_groups.h>
#include <iostream>
#include <stdlib.h>

using namespace std;
namespace cg = cooperative_groups;

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
   Hard coded limit to size of decomposed filter of 16384 floats = 128 KB
 ******************************************************************************/
__constant__ float const_filter[1 << 14];


template<int tile_sz>
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
 * Decomposition, and a Tensor Rank Decomposition.
 *******************************************************************************/
template<unsigned CHANNEL_DIM, unsigned RANK_DIM>
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

  const int n  = blockIdx.z;
  const int Br = Rank / RANK_DIM;

  int tc = (threadIdx.x) % CHANNEL_DIM;
  int tr = (threadIdx.x / CHANNEL_DIM) % Br;
  int w  = (threadIdx.x / CHANNEL_DIM / Br) % Bw;
  int h  = (threadIdx.x / CHANNEL_DIM / Br / Bw);

  const int wOffset = blockIdx.x * Bw;
  const int hOffset = blockIdx.y * Bh;
  int       rOffset = RANK_DIM * tr;

  if (h + hOffset >= H) return;
  if (w + wOffset >= W) return;
  /* if (tr >= Rank) return; */

  float local_acc = 0.0f;
  float rank_acc[RANK_DIM];

#pragma unroll
  for (int r = 0; r < RANK_DIM; ++r) rank_acc[r] = 0.0f;

  for (int c = tc; c < C; c += CHANNEL_DIM) {

    auto active = cg::coalesced_threads();

    const float* iPtr      = Input + n * C * H * W + c * H * W;
    float        local_pix = 0.0f;

    for (int y = 0; y < Y; ++y) {
      for (int x = 0; x < X; ++x) {
        if (h + hOffset + y >= pad       //
            && h + hOffset + y < H + pad //
            && w + wOffset + x >= pad    //
            && w + wOffset + x < W + pad) {

#pragma unroll
          for (int r = 0; r < RANK_DIM; ++r) {
            local_pix += iPtr[(h + hOffset + y - pad) * H //
                              + (w + wOffset + x - pad)]
                         * FX[x * Rank + rOffset + r] //
                         * FY[y * Rank + rOffset + r];
          }
        }
      }
    }

#pragma unroll
    for (int r = 0; r < RANK_DIM; ++r)
      rank_acc[r] += local_pix * FC[tc * Rank + rOffset + r];
  }

  float* work_mem = shared_mem + h * Bw * Br + w * Br;

  local_acc = reduce_sum_tile_shfl<CHANNEL_DIM>(ChannelWarp, local_acc);

  // Save intermediate rank vector to shared memory
  if (ChannelWarp.thread_rank() == 0) work_mem[tr] = local_acc;
  __syncthreads();

  // Swap which threads are in the warp.
  tc = (threadIdx.x / Br) % CHANNEL_DIM;
  tr = threadIdx.x % Br;

  rOffset = RANK_DIM * tr;

  auto RankWarp1 = cg::tiled_partition<1>(cg::this_thread_block());
  auto RankWarp2 = cg::tiled_partition<2>(cg::this_thread_block());
  auto RankWarp4 = cg::tiled_partition<4>(cg::this_thread_block());

#pragma unroll
  for (int r = 0; r < RANK_DIM; ++r) rank_acc[r] = work_mem[r];

  // parallel scale all output channels and sum over rank
  for (int t = tc; t < T; t += CHANNEL_DIM) {

    float output_acc = 0.0;

#pragma unroll
    for (int r = 0; r < RANK_DIM; ++r)
      output_acc += rank_acc[r] * FT[t * Rank + rOffset + r];

    if (Br == 1) output_acc = reduce_sum_tile_shfl<1>(RankWarp1, output_acc);
    if (Br == 2) output_acc = reduce_sum_tile_shfl<2>(RankWarp2, output_acc);
    if (Br == 4) output_acc = reduce_sum_tile_shfl<4>(RankWarp4, output_acc);

    // Output result to global memory
    if (tr == 0) Out[n * T * H * W + tr * H * W + h * W + w] = output_acc;
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

  const unsigned CHANNEL_DIM = 4;
  const unsigned RANK_DIM    = min(4, Rank);
  const unsigned Br          = Rank / RANK_DIM;
  const unsigned Bh          = 4;
  const unsigned Bw          = 4;
  const unsigned BlockSize   = CHANNEL_DIM * Br * Bh * Bw;
  size_t         smsz        = Br * Bh * Bw * sizeof(float);

  cout << BlockSize << " " << CHANNEL_DIM << " "                   //
       << RANK_DIM << " " << Br << " " << Bh << " " << Bw << endl; //"\t\t";

  if (smsz > prop.sharedMemPerBlock) {
    cerr << "Shared Mem Too Big! " << smsz << " > " << prop.sharedMemPerBlock
         << endl;
  }

  const unsigned WgrdDim = (W / Bw) + ((W % Bw) != 0);
  const unsigned HgrdDim = (H / Bh) + ((H % Bh) != 0);
  /* cout << WgrdDim << " " << HgrdDim << endl; */

  const dim3 Gshp(WgrdDim, HgrdDim, N);
  const dim3 Bshp(BlockSize);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float us = 0.0f;

  for (unsigned i = 0; i < PROFCOUNT; ++i) {
    ErrChk(cudaDeviceSynchronize());
    cudaEventRecord(start);

    // clang-format off
    switch(CHANNEL_DIM) {
      /* case 32: */
      /*   switch(RANK_DIM) { */
      /*     case 4: conv2d_cp4_kernel<32,4><<<Gshp, Bshp, smsz>>>(Out, In, H, W, pad, offT, offC, offY, offX, T, C, Y, X, Rank, Bh, Bw); break; */
      /*     case 2: conv2d_cp4_kernel<32,2><<<Gshp, Bshp, smsz>>>(Out, In, H, W, pad, offT, offC, offY, offX, T, C, Y, X, Rank, Bh, Bw); break; */
      /*     case 1: conv2d_cp4_kernel<32,1><<<Gshp, Bshp, smsz>>>(Out, In, H, W, pad, offT, offC, offY, offX, T, C, Y, X, Rank, Bh, Bw); break; */
      /*   } */
      /*          break; */
      /* case 16: */
      /*   switch(RANK_DIM) { */
      /*     case 4: conv2d_cp4_kernel<16,4><<<Gshp, Bshp, smsz>>>(Out, In, H, W, pad, offT, offC, offY, offX, T, C, Y, X, Rank, Bh, Bw); break; */
      /*     case 2: conv2d_cp4_kernel<16,2><<<Gshp, Bshp, smsz>>>(Out, In, H, W, pad, offT, offC, offY, offX, T, C, Y, X, Rank, Bh, Bw); break; */
      /*     case 1: conv2d_cp4_kernel<16,1><<<Gshp, Bshp, smsz>>>(Out, In, H, W, pad, offT, offC, offY, offX, T, C, Y, X, Rank, Bh, Bw); break; */
      /*   } */
      /*          break; */
      /* case 8: */
      /*   switch(RANK_DIM) { */
      /*     case 4: conv2d_cp4_kernel<8,4><<<Gshp, Bshp, smsz>>>(Out, In, H, W, pad, offT, offC, offY, offX, T, C, Y, X, Rank, Bh, Bw); break; */
      /*     case 2: conv2d_cp4_kernel<8,2><<<Gshp, Bshp, smsz>>>(Out, In, H, W, pad, offT, offC, offY, offX, T, C, Y, X, Rank, Bh, Bw); break; */
      /*     case 1: conv2d_cp4_kernel<8,1><<<Gshp, Bshp, smsz>>>(Out, In, H, W, pad, offT, offC, offY, offX, T, C, Y, X, Rank, Bh, Bw); break; */
      /*   } */
      /*          break; */
      case 4:
        switch(RANK_DIM) {
          case 4: conv2d_cp4_kernel<4,4><<<Gshp, Bshp, smsz>>>(Out, In, H, W, pad, offT, offC, offY, offX, T, C, Y, X, Rank, Bh, Bw); break;
          case 2: conv2d_cp4_kernel<4,2><<<Gshp, Bshp, smsz>>>(Out, In, H, W, pad, offT, offC, offY, offX, T, C, Y, X, Rank, Bh, Bw); break;
          case 1: conv2d_cp4_kernel<4,1><<<Gshp, Bshp, smsz>>>(Out, In, H, W, pad, offT, offC, offY, offX, T, C, Y, X, Rank, Bh, Bw); break;
        }
               break;
      /* case 2: */
      /*   switch(RANK_DIM) { */
      /*     case 4: conv2d_cp4_kernel<2,4><<<Gshp, Bshp, smsz>>>(Out, In, H, W, pad, offT, offC, offY, offX, T, C, Y, X, Rank, Bh, Bw); break; */
      /*     case 2: conv2d_cp4_kernel<2,2><<<Gshp, Bshp, smsz>>>(Out, In, H, W, pad, offT, offC, offY, offX, T, C, Y, X, Rank, Bh, Bw); break; */
      /*     case 1: conv2d_cp4_kernel<2,1><<<Gshp, Bshp, smsz>>>(Out, In, H, W, pad, offT, offC, offY, offX, T, C, Y, X, Rank, Bh, Bw); break; */
      /*   } */
      /*          break; */
      /* case 1: */
      /*   switch(RANK_DIM) { */
      /*     case 4: conv2d_cp4_kernel<1,4><<<Gshp, Bshp, smsz>>>(Out, In, H, W, pad, offT, offC, offY, offX, T, C, Y, X, Rank, Bh, Bw); break; */
      /*     case 2: conv2d_cp4_kernel<1,2><<<Gshp, Bshp, smsz>>>(Out, In, H, W, pad, offT, offC, offY, offX, T, C, Y, X, Rank, Bh, Bw); break; */
      /*     case 1: conv2d_cp4_kernel<1,1><<<Gshp, Bshp, smsz>>>(Out, In, H, W, pad, offT, offC, offY, offX, T, C, Y, X, Rank, Bh, Bw); break; */
      /*   } */
      /*          break; */
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
