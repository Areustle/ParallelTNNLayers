#include "conv.cuh"

__constant__ float const_filter[4096];

// The Full convolution kernel.
template<unsigned TileFactor = 1>
__global__ void conv2d_full_kernel(const float* __restrict__ Input,
                                   const unsigned N,
                                   const unsigned C,
                                   const unsigned H,
                                   const unsigned W,
                                   const unsigned pad,
                                   const unsigned fK,
                                   const unsigned fH,
                                   const unsigned fW,
                                   float* __restrict__ Out) {

  extern __shared__ float shared_mem[];

  // Declare useful constants. This should be cleaned up if
  // Register pressure grows too high.
  const unsigned w         = threadIdx.x;
  const unsigned h         = threadIdx.y;
  const unsigned Bw        = blockDim.x;
  const unsigned Bh        = blockDim.y;
  const unsigned wBlockOff = blockIdx.x * blockDim.x * TileFactor;
  const unsigned hBlockOff = blockIdx.y * blockDim.y;
  const unsigned jEnd      = fH - 1 + Bh;
  const unsigned iEnd      = fW - 1 + Bw;
  const unsigned sH        = fH - 1 + Bh;
  const unsigned sW        = fH - 1 + Bw * TileFactor;

  // Grid Stride loop to handle overlarge batch (n) and filter (k) sizes
  for (int n = blockIdx.z / fK; n < N; n += blockDim.z * gridDim.z) {
    for (int k = blockIdx.z % fK; k < fK; k += blockDim.z * gridDim.z) {

      // Shift the Global pointers to our Region Of interest
      const float* iPtr =
          Input + n * C * H * W; // batch number offset for this thread

      float* oPtr = Out + n * fK * H * W // batch offset
                    + k * H * W          // conv filter offset
                    + hBlockOff * W      // h offset
                    + wBlockOff;         // w offset

      // clang-format off
      // Cooperatively load all input segment into our shared memory and pad it.
      for (unsigned c = 0; c < C; ++c)         // For every channel
      for (unsigned j = h; j < jEnd; j += Bh)  // For every participating h pixel
      for (unsigned i = w; i < iEnd; i += Bw)  // For every participating w pixel
      #pragma unroll
      for (unsigned t = 0; t < TileFactor; ++t)
        shared_mem[c*sH*sW + j*sW + i+(t*Bw)]
          = (j+hBlockOff >= pad
              && j+hBlockOff < H+pad
              && i+wBlockOff >= pad
              && i+wBlockOff+(t*Bw) < W+pad)
          ?(iPtr[c*H*W                          // Channel
                  + (j+hBlockOff-pad)*W         // Height
                  + (i+wBlockOff-pad)+(t*Bw)])  // Width
          :(0.0f); // Pad with Zeros if outside the bounds

      __syncthreads();

      // Handle block / input size mismatch. This occurs here and not earlier
      // So that these threads can still participate in the cooperative shared
      // Memory load.
      if (hBlockOff + h >= H) continue;
      if (wBlockOff + w >= W) continue;

      // Build sum by tiling factor. If tiling factor is >1 then each thread
      // will calculate multiple output pixels in local registers.
      float sum[TileFactor];
      #pragma unroll
      for (unsigned t = 0; t < TileFactor; ++t) sum[t] = 0.0f;

      // Perform Convolution from shared memory.
      // Accumulate sum of products in 'sum' variable for each t.
      // currently expect this to have bank conflicts. Requires padding.
      for (unsigned c = 0; c < C; ++c)
      for (unsigned r = 0; r < fH; ++r)
      for (unsigned s = 0; s < fW; ++s)
      #pragma unroll
      for (unsigned t = 0; t < TileFactor; ++t)
        sum[t] += shared_mem[c*sH*sW + (h+r)*sW + (w+s+(t*Bw))]
          * const_filter[k*C*fH*fW + c*fH*fW + r*fW + s];

      // populate output array.
      #pragma unroll
      for (unsigned t = 0; t < TileFactor; ++t)
        oPtr[h*W + w+(t*Bw)] = sum[t];
      // clang-format on
    }
  }
}

void cuda_conv2d_full_gpu(const float* In,
                          const int    N,
                          const int    C,
                          const int    H,
                          const int    W,
                          const int    pad,
                          const float* Filter,
                          const int    fK,
                          const int    fC,
                          const int    fH,
                          const int    fW,
                          float*       Out) {

  // This implementation uses the GPU's constant memory as a fast cache to
  // hold the relatively small and unchanging filter weights. These must all
  // be accessed uniformly by the threads in a block for parallel execution.
  cudaMemcpyToSymbol(const_filter, Filter, sizeof(float) * fK * fC * fH * fW);

  const unsigned        hdim = 16;
  const unsigned        wdim = 16;
  static const unsigned tf   = 1;
  const size_t          smsz =
      C //
        // If using shared memory padding to avoid bank conflicts
      /* * ((((fW - 1) / 32) + ((fW - 1) % 32 != 0)) * 32 + bdim * tf) // */
      * (fH - 1 + wdim * tf) //
      * (fH - 1 + hdim)      //
      * sizeof(float);

  const unsigned WgrdDim = (W / (wdim * tf)) + ((W % (wdim * tf)) != 0);
  const unsigned HgrdDim = (H / hdim) + ((H % hdim) != 0);
  const dim3     Gshp(WgrdDim, HgrdDim, fK * N);
  const dim3     Bshp(wdim, hdim, 1);

  conv2d_full_kernel<tf>
      <<<Gshp, Bshp, smsz>>>(In, N, C, H, W, pad, fK, fH, fW, Out);
  cudaDeviceSynchronize();
}

Tensor conv2d_full_gpu(Tensor const Input, Tensor const Filter, int pad) {

  const int N  = Input.shape[0];
  const int C  = Input.shape[1];
  const int H  = Input.shape[2];
  const int W  = Input.shape[3];
  const int fK = Filter.shape[0];
  const int fC = Filter.shape[1];
  const int fH = Filter.shape[2];
  const int fW = Filter.shape[3];
  Tensor    Output{ N, fK, H, W };
  cuda_conv2d_full_gpu(Input.m_data,
                       N,
                       C,
                       H,
                       W,
                       pad,
                       Filter.m_data,
                       fK,
                       fC,
                       fH,
                       fW,
                       Output.m_data);
  return Output;
}

Tensor conv2d_full_cpu(Tensor const Input, Tensor const Filter) {

  const int N  = Input.shape[0];
  const int C  = Input.shape[1];
  const int H  = Input.shape[2];
  const int W  = Input.shape[3];
  const int FK = Filter.shape[0];
  const int FC = Filter.shape[1];
  const int FR = Filter.shape[2];
  const int FS = Filter.shape[3];

  const int FRCenter = FR / 2;
  const int FSCenter = FS / 2;

  Tensor Out{ N, C, H, W };

  // clang-format off
  for (int n = 0; n < N; ++n)
  for (int fk = 0; fk < FK; ++fk)
  for (int h = 0; h < H; ++h)
  for (int w = 0; w < W; ++w){
    float sum = 0.0f;
    for (int c = 0; c < C; ++c)
    for (int fr = 0; fr < FR; ++fr)
    for (int fs = 0; fs < FS; ++fs){

      const int hIdx = h + (fr - FRCenter);
      const int wIdx = w + (fs - FSCenter);

      if(hIdx >= 0 && hIdx < H && wIdx >= 0 && wIdx < W){
            sum += Input.m_data[n*C*H*W + c*H*W + hIdx*W + wIdx]
            *  Filter.m_data[fk*C*FR*FS + c*FR*FS + fr*FS + fs];
      }

    }
    Out.m_data[n*C*H*W + fk*H*W + h*W + w] = sum;
  }
  // clang-format on

  return Out;
}
