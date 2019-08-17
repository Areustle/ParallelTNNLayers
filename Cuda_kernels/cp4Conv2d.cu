#include "cp4Conv2d.cuh"
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
                                  const unsigned offK,
                                  const unsigned offC,
                                  const unsigned offH,
                                  const unsigned offW,
                                  const unsigned fK,
                                  const unsigned sW,
                                  const unsigned sH) {

  extern __shared__ float shared_mem[];

  const unsigned w         = threadIdx.x;
  const unsigned h         = threadIdx.y;
  const unsigned wBlockOff = blockIdx.x * blockDim.x;
  const unsigned hBlockOff = blockIdx.y * blockDim.y;
  const unsigned n         = blockIdx.z;

  float local_pixel_acc[Rank];
  for (unsigned r = 0; r < Rank; ++r) local_pixel_acc[r] = 0.0f;

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

    float tmpxl[Rank];

    for (unsigned r = 0; r < Rank; ++r) tmpxl[r] = 0.0f;

    for (unsigned fh = 0; fh < FilterDim; ++fh)
      for (unsigned fw = 0; fw < FilterDim; ++fw)
#pragma unroll
        for (unsigned r = 0; r < Rank; ++r)
          tmpxl[r] += shared_mem[(h + fh) * sW + (w + fw)]
                      * const_filter[offH + fh * Rank + r]
                      * const_filter[offW + fw * Rank + r];

    for (unsigned r = 0; r < Rank; ++r)
      local_pixel_acc[r] += tmpxl[r] * const_filter[offC + c * Rank + r];

    __syncthreads();
  }

  if (hBlockOff + h >= H) return;
  if (wBlockOff + w >= W) return;

  /****************************************************************************
   * Reduce over rank while scaling by kth filter value.
   ****************************************************************************/
  for (unsigned k = 0; k < fK; ++k) {

    float kth_filter_pixel = 0.0f;

    for (unsigned r = 0; r < Rank; ++r)
      kth_filter_pixel
          += local_pixel_acc[r] * const_filter[offK + 0 * Rank + r];

    Out[n * fK * H * W + k * H * W + (h + hBlockOff) * W + w + wBlockOff]
        = kth_filter_pixel;
  }
}


float CP4Conv2dGPU(tensor_shape params,
                   const float* In,
                   const float* FilterK,
                   const float* FilterC,
                   const float* FilterH,
                   const float* FilterW,
                   float*       Out,
                   unsigned     PROFCOUNT = 1) {

  const unsigned N     = params.N;
  const unsigned C     = params.C;
  const unsigned H     = params.H;
  const unsigned W     = params.W;
  const unsigned pad   = params.pad;
  const unsigned fRank = params.fRank;
  const unsigned fK    = params.fK;
  const unsigned fC    = params.C;
  const unsigned fH    = params.fH;
  const unsigned fW    = params.fW;

  if (fH != fW) cerr << "Invalid filter shape. Height must equal width" << endl;

  // This implementation uses the GPU's constant memory as a fast cache to
  // hold the relatively small and unchanging filter weights. These must all
  // be accessed uniformly by the threads in a block for parallel execution.
  // Populate GPU constant memory with the 4 filters at an appropriate offset.
  const unsigned offK = 0;
  const unsigned offC = offK + (fK * fRank);
  const unsigned offH = offC + (fC * fRank);
  const unsigned offW = offH + (fH * fRank);
  ErrChk(cudaMemcpyToSymbol(const_filter,
                            FilterK,
                            sizeof(float) * (fK * fRank),
                            sizeof(float) * offK));
  ErrChk(cudaMemcpyToSymbol(const_filter,
                            FilterC,
                            sizeof(float) * (fC * fRank),
                            sizeof(float) * offC));
  ErrChk(cudaMemcpyToSymbol(const_filter,
                            FilterH,
                            sizeof(float) * (fH * fRank),
                            sizeof(float) * offH));
  ErrChk(cudaMemcpyToSymbol(const_filter,
                            FilterW,
                            sizeof(float) * (fW * fRank),
                            sizeof(float) * offW));

  cudaDeviceProp prop;
  ErrChk(cudaGetDeviceProperties(&prop, 0));

  unsigned Bh   = 4;
  unsigned Bw   = 16;
  unsigned sW   = fW - 1 + Bw;
  unsigned sH   = fH - 1 + Bh;
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
  float cumulativeNS = 0.0f;

  for (unsigned i = 0; i < PROFCOUNT; ++i) {
    ErrChk(cudaDeviceSynchronize());
    cudaEventRecord(start);
    // clang-format off
    switch (fW) {
      case 1:
        switch (fRank) {
          case  1: conv2d_cp4_kernel< 1, 1><<<Gshp, Bshp, smsz>>>(Out, In, N, C, H, W, pad, offK, offC, offH, offW, fK, sW, sH); break;
          case  2: conv2d_cp4_kernel< 1, 2><<<Gshp, Bshp, smsz>>>(Out, In, N, C, H, W, pad, offK, offC, offH, offW, fK, sW, sH); break;
          case  4: conv2d_cp4_kernel< 1, 4><<<Gshp, Bshp, smsz>>>(Out, In, N, C, H, W, pad, offK, offC, offH, offW, fK, sW, sH); break;
          case  8: conv2d_cp4_kernel< 1, 8><<<Gshp, Bshp, smsz>>>(Out, In, N, C, H, W, pad, offK, offC, offH, offW, fK, sW, sH); break;
          case 16: conv2d_cp4_kernel< 1,16><<<Gshp, Bshp, smsz>>>(Out, In, N, C, H, W, pad, offK, offC, offH, offW, fK, sW, sH); break;
          default: cerr << "Rank not supported!" << endl;
        }
        break;
      case 3:
        switch (fRank) {
          case  1: conv2d_cp4_kernel< 3, 1><<<Gshp, Bshp, smsz>>>(Out, In, N, C, H, W, pad, offK, offC, offH, offW, fK, sW, sH); break;
          case  2: conv2d_cp4_kernel< 3, 2><<<Gshp, Bshp, smsz>>>(Out, In, N, C, H, W, pad, offK, offC, offH, offW, fK, sW, sH); break;
          case  4: conv2d_cp4_kernel< 3, 4><<<Gshp, Bshp, smsz>>>(Out, In, N, C, H, W, pad, offK, offC, offH, offW, fK, sW, sH); break;
          case  8: conv2d_cp4_kernel< 3, 8><<<Gshp, Bshp, smsz>>>(Out, In, N, C, H, W, pad, offK, offC, offH, offW, fK, sW, sH); break;
          case 16: conv2d_cp4_kernel< 3,16><<<Gshp, Bshp, smsz>>>(Out, In, N, C, H, W, pad, offK, offC, offH, offW, fK, sW, sH); break;
          default: cerr << "Rank not supported!" << endl;
        }
        break;
      case 5:
        switch (fRank) {
          case  1: conv2d_cp4_kernel< 5, 1><<<Gshp, Bshp, smsz>>>(Out, In, N, C, H, W, pad, offK, offC, offH, offW, fK, sW, sH); break;
          case  2: conv2d_cp4_kernel< 5, 2><<<Gshp, Bshp, smsz>>>(Out, In, N, C, H, W, pad, offK, offC, offH, offW, fK, sW, sH); break;
          case  4: conv2d_cp4_kernel< 5, 4><<<Gshp, Bshp, smsz>>>(Out, In, N, C, H, W, pad, offK, offC, offH, offW, fK, sW, sH); break;
          case  8: conv2d_cp4_kernel< 5, 8><<<Gshp, Bshp, smsz>>>(Out, In, N, C, H, W, pad, offK, offC, offH, offW, fK, sW, sH); break;
          case 16: conv2d_cp4_kernel< 5,16><<<Gshp, Bshp, smsz>>>(Out, In, N, C, H, W, pad, offK, offC, offH, offW, fK, sW, sH); break;
          default: cerr << "Rank not supported!" << endl;
        }
        break;
      case 7:
        switch (fRank) {
          case  1: conv2d_cp4_kernel< 7, 1><<<Gshp, Bshp, smsz>>>(Out, In, N, C, H, W, pad, offK, offC, offH, offW, fK, sW, sH); break;
          case  2: conv2d_cp4_kernel< 7, 2><<<Gshp, Bshp, smsz>>>(Out, In, N, C, H, W, pad, offK, offC, offH, offW, fK, sW, sH); break;
          case  4: conv2d_cp4_kernel< 7, 4><<<Gshp, Bshp, smsz>>>(Out, In, N, C, H, W, pad, offK, offC, offH, offW, fK, sW, sH); break;
          case  8: conv2d_cp4_kernel< 7, 8><<<Gshp, Bshp, smsz>>>(Out, In, N, C, H, W, pad, offK, offC, offH, offW, fK, sW, sH); break;
          case 16: conv2d_cp4_kernel< 7,16><<<Gshp, Bshp, smsz>>>(Out, In, N, C, H, W, pad, offK, offC, offH, offW, fK, sW, sH); break;
          default: cerr << "Rank not supported!" << endl;
        }
        break;
      case 9:
        switch (fRank) {
          case  1: conv2d_cp4_kernel< 9, 1><<<Gshp, Bshp, smsz>>>(Out, In, N, C, H, W, pad, offK, offC, offH, offW, fK, sW, sH); break;
          case  2: conv2d_cp4_kernel< 9, 2><<<Gshp, Bshp, smsz>>>(Out, In, N, C, H, W, pad, offK, offC, offH, offW, fK, sW, sH); break;
          case  4: conv2d_cp4_kernel< 9, 4><<<Gshp, Bshp, smsz>>>(Out, In, N, C, H, W, pad, offK, offC, offH, offW, fK, sW, sH); break;
          case  8: conv2d_cp4_kernel< 9, 8><<<Gshp, Bshp, smsz>>>(Out, In, N, C, H, W, pad, offK, offC, offH, offW, fK, sW, sH); break;
          case 16: conv2d_cp4_kernel< 9,16><<<Gshp, Bshp, smsz>>>(Out, In, N, C, H, W, pad, offK, offC, offH, offW, fK, sW, sH); break;
          default: cerr << "Rank not supported!" << endl;
        }
        break;
      case 11:
        switch (fRank) {
          case  1: conv2d_cp4_kernel<11, 1><<<Gshp, Bshp, smsz>>>(Out, In, N, C, H, W, pad, offK, offC, offH, offW, fK, sW, sH); break;
          case  2: conv2d_cp4_kernel<11, 2><<<Gshp, Bshp, smsz>>>(Out, In, N, C, H, W, pad, offK, offC, offH, offW, fK, sW, sH); break;
          case  4: conv2d_cp4_kernel<11, 4><<<Gshp, Bshp, smsz>>>(Out, In, N, C, H, W, pad, offK, offC, offH, offW, fK, sW, sH); break;
          case  8: conv2d_cp4_kernel<11, 8><<<Gshp, Bshp, smsz>>>(Out, In, N, C, H, W, pad, offK, offC, offH, offW, fK, sW, sH); break;
          case 16: conv2d_cp4_kernel<11,16><<<Gshp, Bshp, smsz>>>(Out, In, N, C, H, W, pad, offK, offC, offH, offW, fK, sW, sH); break;
          default: cerr << "Rank not supported!" << endl;
        }
        break;
      case 13:
        switch (fRank) {
          case  1: conv2d_cp4_kernel<13, 1><<<Gshp, Bshp, smsz>>>(Out, In, N, C, H, W, pad, offK, offC, offH, offW, fK, sW, sH); break;
          case  2: conv2d_cp4_kernel<13, 2><<<Gshp, Bshp, smsz>>>(Out, In, N, C, H, W, pad, offK, offC, offH, offW, fK, sW, sH); break;
          case  4: conv2d_cp4_kernel<13, 4><<<Gshp, Bshp, smsz>>>(Out, In, N, C, H, W, pad, offK, offC, offH, offW, fK, sW, sH); break;
          case  8: conv2d_cp4_kernel<13, 8><<<Gshp, Bshp, smsz>>>(Out, In, N, C, H, W, pad, offK, offC, offH, offW, fK, sW, sH); break;
          case 16: conv2d_cp4_kernel<13,16><<<Gshp, Bshp, smsz>>>(Out, In, N, C, H, W, pad, offK, offC, offH, offW, fK, sW, sH); break;
          default: cerr << "Rank not supported!" << endl;
        }
        break;
      case 15:
        switch (fRank) {
          case  1: conv2d_cp4_kernel<15, 1><<<Gshp, Bshp, smsz>>>(Out, In, N, C, H, W, pad, offK, offC, offH, offW, fK, sW, sH); break;
          case  2: conv2d_cp4_kernel<15, 2><<<Gshp, Bshp, smsz>>>(Out, In, N, C, H, W, pad, offK, offC, offH, offW, fK, sW, sH); break;
          case  4: conv2d_cp4_kernel<15, 4><<<Gshp, Bshp, smsz>>>(Out, In, N, C, H, W, pad, offK, offC, offH, offW, fK, sW, sH); break;
          case  8: conv2d_cp4_kernel<15, 8><<<Gshp, Bshp, smsz>>>(Out, In, N, C, H, W, pad, offK, offC, offH, offW, fK, sW, sH); break;
          case 16: conv2d_cp4_kernel<15,16><<<Gshp, Bshp, smsz>>>(Out, In, N, C, H, W, pad, offK, offC, offH, offW, fK, sW, sH); break;
          default: cerr << "Rank not supported!" << endl;
        }
        break;
      case 17:
        switch (fRank) {
          case  1: conv2d_cp4_kernel<17, 1><<<Gshp, Bshp, smsz>>>(Out, In, N, C, H, W, pad, offK, offC, offH, offW, fK, sW, sH); break;
          case  2: conv2d_cp4_kernel<17, 2><<<Gshp, Bshp, smsz>>>(Out, In, N, C, H, W, pad, offK, offC, offH, offW, fK, sW, sH); break;
          case  4: conv2d_cp4_kernel<17, 4><<<Gshp, Bshp, smsz>>>(Out, In, N, C, H, W, pad, offK, offC, offH, offW, fK, sW, sH); break;
          case  8: conv2d_cp4_kernel<17, 8><<<Gshp, Bshp, smsz>>>(Out, In, N, C, H, W, pad, offK, offC, offH, offW, fK, sW, sH); break;
          case 16: conv2d_cp4_kernel<17,16><<<Gshp, Bshp, smsz>>>(Out, In, N, C, H, W, pad, offK, offC, offH, offW, fK, sW, sH); break;
          default: cerr << "Rank not supported!" << endl;
        }
        break;
      default: cerr << "Filter shape not supported!" << endl;
    }
  
    cudaEventRecord(stop);
    // clang-format on
    ErrChk(cudaPeekAtLastError());
    ErrChk(cudaDeviceSynchronize());

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cumulativeNS += milliseconds * 1e6;
  }
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return cumulativeNS / PROFCOUNT;
}


/*******************************************************************************
 * Unified memory Tensorized call of Convolution in GPU
 ******************************************************************************/
Tensor CP::Conv2dForward(Tensor const Input,
                         Tensor const FilterK,
                         Tensor const FilterC,
                         Tensor const FilterH,
                         Tensor const FilterW,
                         unsigned     pad) {

  tensor_shape params;
  params.N     = Input.shape[0];
  params.C     = Input.shape[1];
  params.H     = Input.shape[2];
  params.W     = Input.shape[3];
  params.pad   = pad;
  params.fRank = FilterK.shape[1];
  params.fK    = FilterK.shape[0];
  params.fH    = FilterH.shape[0];
  params.fW    = FilterW.shape[0];

  Tensor Out{ params.N, params.fK, params.H, params.W };

  CP4Conv2dGPU(params,
               Input.m_data,
               FilterK.m_data,
               FilterC.m_data,
               FilterH.m_data,
               FilterW.m_data,
               Out.m_data);

  return Out;
}


/*******************************************************************************
 * Run_convolution operation with a profile count loop
 ******************************************************************************/
float CP::run_convolution(tensor_shape p, unsigned PROFCOUNT) {

  float* In;
  float* Out;
  float* FilterK;
  float* FilterC;
  float* FilterW;
  float* FilterH;

  cudaMalloc(&In, p.N * p.C * p.H * p.W * sizeof(float));
  cudaMalloc(&FilterK, p.fK * p.fRank * sizeof(float));
  cudaMalloc(&FilterC, p.C * p.fRank * sizeof(float));
  cudaMalloc(&FilterH, p.fH * p.fRank * sizeof(float));
  cudaMalloc(&FilterW, p.fW * p.fRank * sizeof(float));
  cudaMalloc(&Out, p.N * p.fK * p.H * p.W * sizeof(float));


  float ns
      = CP4Conv2dGPU(p, In, FilterK, FilterC, FilterH, FilterW, Out, PROFCOUNT);

  cudaFree(In);
  cudaFree(FilterK);
  cudaFree(FilterC);
  cudaFree(FilterH);
  cudaFree(FilterW);
  cudaFree(Out);

  return ns;
}


/*******************************************************************************
 * Main function. call 1 instance of kernel execution
 ******************************************************************************/
int main(int argc, char** argv) {

  unsigned N     = 5;
  unsigned C     = 32;
  unsigned H     = 1024;
  unsigned W     = 1024;
  unsigned pad   = 1;
  unsigned fK    = 32;
  unsigned fH    = 3;
  unsigned fW    = 3;
  unsigned fRank = 8;

  if (argc != 11) {
    cerr << "Using Default shape" << endl;
    cudaSetDevice(0);
  } else {
    N     = atoi(argv[1]);
    C     = atoi(argv[2]);
    H     = atoi(argv[3]);
    W     = atoi(argv[4]);
    pad   = atoi(argv[5]);
    fK    = atoi(argv[6]);
    fH    = atoi(argv[7]);
    fW    = atoi(argv[8]);
    fRank = atoi(argv[9]);
    cudaSetDevice(atoi(argv[10]));
  }

  tensor_shape params;
  params.N     = N;
  params.C     = C;
  params.H     = H;
  params.W     = W;
  params.pad   = pad;
  params.fRank = fRank;
  params.fK    = fK;
  params.fH    = fH;
  params.fW    = fW;

  CP::run_convolution(params, 1);
}
