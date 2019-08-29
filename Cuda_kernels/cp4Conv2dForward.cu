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
    if (abort)
      exit(code);
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
__global__ void conv2d_cp4_kernel(float* __restrict__ Out,
                                  const float* __restrict__ Input,
                                  const unsigned C,
                                  const unsigned H,
                                  const unsigned W,
                                  const unsigned pad,
                                  const unsigned offT,
                                  const unsigned offC,
                                  const unsigned offY,
                                  const unsigned offX,
                                  const unsigned T,
                                  const unsigned Y,
                                  const unsigned X,
                                  const unsigned Rank,
                                  const unsigned Bw,
                                  const unsigned Bh,
                                  const unsigned sW,
                                  const unsigned sH) {

  extern __shared__ float shared_mem[];

  const unsigned h         = threadIdx.z / Bw;
  const unsigned w         = threadIdx.z % Bw;
  const unsigned n         = blockIdx.z;
  const unsigned wBlockOff = blockIdx.x * Bw;
  const unsigned hBlockOff = blockIdx.y * Bh;

  unsigned r = threadIdx.y;

  float local_acc = 0.0f;

  // Parallel multiply by channel and rank.
  for (unsigned c = threadIdx.x; c < C; c += blockDim.x) {
    // Shift the Global pointers to our Region Of interest
    const float* iPtr = Input + n * C * H * W + c * H * W;

    // Cooperatively load all input segment into our shared memory and pad it.
    for (unsigned hh = h; hh < sH; hh += Bh)
      for (unsigned ww = w; ww < sW; ww += Bw)
        shared_mem[c * sH * sW * Rank + hh * sW * Rank + ww * Rank + r] =
            (hh + hBlockOff >= pad       //
             && hh + hBlockOff < H + pad //
             && ww + wBlockOff >= pad    //
             && ww + wBlockOff < W + pad)
                ? iPtr[(hh + hBlockOff - pad) * W + (ww + wBlockOff - pad)]
                : (0.0f); // Pad with Zeros if outside the bounds

    __syncthreads();

    float local_pix = 0.0f;

    for (unsigned y = 0; y < Y; ++y) {
      local_x = 0.0f;
      for (unsigned x = 0; x < X; ++x) {
        local_x += shared_mem[(h + y) * sW + (w + x)]
                   * const_filter[offX + x * Rank + r];
      }
      local_pix += local_x * const_filter[offY + y * Rank + r];
    }
    local_acc += local_pix * const_filter[offC + c * Rank + r];
  }

  // Handle block / input size mismatch. This occurs here and not earlier
  // So that these threads can still participate in the cooperative shared
  // Memory load.
  if (hBlockOff + h >= H) return;
  if (wBlockOff + w >= W) return;

  // Store intermediate result in shared memory
  __syncthreads();
  shared_mem[r * blockDim.x + threadIdx.x] = local_acc;
  __syncthreads();

  // Reduce over channels for a given rank in shared memory.
  for (unsigned cc = ((blockDim.x >> 1) << 1); cc > 0; cc >>= 1) {
    if (threadIdx.x < cc && (threadIdx.x + cc) < C)
      shared_mem[r * blockDim.x + threadIdx.x] +=
          shared_mem[r * blockDim.x + (threadIdx.x + cc)];
    __syncthreads();
  }

  local_acc = shared_mem[r * blockDim.x];
  __syncthreads();
  r = threadIdx.x;


  for (unsigned t = threadIdx.y; t < T; t += blockDim.y) {
    __syncthreads();
    shared_mem[t * Rank + threadIdx.x] =
        local_acc * const_filter[offT + t * Rank + r];
    __syncthreads();

    // Reduce over rank for a given output channel in shared memory.
    for (unsigned rr = ((Rank >> 1) << 1); rr > 0; rr >>= 1) {
      if (threadIdx.x < rr && (threadIdx.x + rr) < Rank)
        shared_mem[t * Rank + threadIdx.x] +=
            shared_mem[t * Rank + (threadIdx.x + cc)];
      __syncthreads();
    }

    // Write this output pixel to global memory
    if (threadIdx.x == 0)
      Out[n * T * H * W + t * H * W + h * W + w] = shared_mem[t * Rank];
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

  if (Y != X)
    cerr << "Invalid filter shape. Height must equal width" << endl;

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

  unsigned Bw = 1;
  unsigned Bh = 1;
  unsigned Bc = 1;
  unsigned sW = X - 1 + Bw;
  unsigned sH = Y - 1 + Bh;
  /* size_t   smsz = sW * sH * sizeof(float); */
  size_t smsz = Bc * sH * sW * Rank * sizeof(float);

  if (smsz > prop.sharedMemPerBlock) {
    cerr << "Shared Mem Too Big! " << smsz << " > " << prop.sharedMemPerBlock
         << endl;
  }

  const unsigned WgrdDim = (W / Bw) + ((W % Bw) != 0);
  const unsigned HgrdDim = (H / Bh) + ((H % Bh) != 0);
  const dim3     Gshp(WgrdDim, HgrdDim, N);
  const dim3     Bshp(Bc, Rank, Bw * Bh);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float us = 0.0f;

  for (unsigned i = 0; i < PROFCOUNT; ++i) {
    ErrChk(cudaDeviceSynchronize());
    cudaEventRecord(start);
    // clang-format off
    conv2d_cp4_kernel<<<Gshp, Bshp, smsz>>>(Out, In, C, H, W, pad, offT, offC, offY, offX, T, Y, X, Rank, Bw, Bh, sW, sH); break;
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
template<unsigned FilterDim, unsigned Rank>
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
  if (hBlockOff + h >= H)
    return;
  if (wBlockOff + w >= W)
    return;

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
