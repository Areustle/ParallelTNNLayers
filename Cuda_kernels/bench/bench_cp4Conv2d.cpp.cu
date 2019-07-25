
__constant__ float const_filter[4096];

// The Full convolution kernel.
template<unsigned TileFactor = 1>
__global__ void conv2d_full_kernel(const float* __restrict__ Input,
                                   const unsigned pad,
                                   const unsigned fK,
                                   const unsigned fH,
                                   const unsigned fW,
                                   const unsigned C,
                                   float* __restrict__ Out) {

  extern __shared__ float shared_mem[];

  // Declare useful constants. This should be cleaned up if
  // Register pressure grows too high.
  const unsigned n         = blockIdx.z / fK;
  const unsigned k         = blockIdx.z % fK;
  const unsigned w         = threadIdx.x;
  const unsigned h         = threadIdx.y;
  const unsigned Bw        = blockDim.x;
  const unsigned Bh        = blockDim.y;
  const unsigned oW        = gridDim.x * blockDim.x * TileFactor;
  const unsigned oH        = gridDim.y * blockDim.y;
  const unsigned iW        = gridDim.x * blockDim.x * TileFactor + pad;
  const unsigned iH        = gridDim.y * blockDim.y + pad;
  const unsigned hBlockOff = blockIdx.y * blockDim.y;
  const unsigned wBlockOff = blockIdx.x * blockDim.x * TileFactor;
  const unsigned jEnd      = fH - 1 + Bh;
  const unsigned iEnd      = fW - 1 + Bw;
  const unsigned sH        = fH - 1 + Bh;
  const unsigned sW        = fW - 1 + Bw * TileFactor;

  // Shift the Global pounsigneders to our Region Of unsignederest
  Input += n * C * iH * iW  // batch number offset for this thread
           + hBlockOff * iW // h offset for this thread
           + wBlockOff;     // w offset for this thread

  Out += n * fK * oH * oW // batch offset
         + k * oH * oW    // conv filter offset
         + hBlockOff * oW // h offset
         + wBlockOff;     // w offset
  // clang-format off

  // Cooperatively load all input segment unsignedo our shared memory.
  for (unsigned c = 0; c < C; ++c)         // For every channel
  for (unsigned j = h; j < jEnd; j += Bh)  // For every participating h pixel
  for (unsigned i = w; i < iEnd; i += Bw)  // For every participating w pixel
  #pragma unroll
  for (unsigned t = 0; t < TileFactor; ++t)
    shared_mem[c*sH*sW + j*sW + i+(t*Bw)] = Input[c*iH*iW + j*iW + i+(t*Bw)];

  __syncthreads();

  // Build sum by tiling factor
  float sum[TileFactor];
  #pragma unroll
  for (unsigned t = 0; t < TileFactor; ++t) sum[t] = 0.0f;

  // Perform Convolution from shared memory
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
    Out[h*oW + w+(t*Bw)] = sum[t];

  // clang-format on
}

int main() {

  float* U;
  float* V;
  void*  K;

  unsigned N  = 1;
  unsigned C  = 16;
  unsigned H  = 32;
  unsigned W  = 32;
  unsigned fK = 16;
  unsigned fH = 3;
  unsigned fW = 3;

  cudaMalloc(&U, (N * C * (H + 2) * (W + 2)) * sizeof(float));
  cudaMalloc(&K, (fK * C * fH * fW) * sizeof(float));
  cudaMalloc(&V, (N * fK * H * W) * sizeof(float));
  cudaMemcpyToSymbol(const_filter, K, sizeof(float) * (C * fK * fH * fW));

  static const unsigned tf   = 2;
  const unsigned        bdim = 16;
  const size_t          smsz = C             //
                      * (fW - 1 + bdim * tf) //
                      * (fH - 1 + bdim) *    //
                      sizeof(float);

  const dim3 Gshp(W / (bdim * tf), H / (bdim), fK * N);
  const dim3 Bshp(bdim, bdim, 1);

  /* for (unsigned i = 0; i < 1000; ++i) { */
  conv2d_full_kernel<tf><<<Gshp, Bshp, smsz>>>(U, 2, fK, fH, fW, C, V);
  cudaDeviceSynchronize();
  /* } */

  cudaFree(U);
  cudaFree(V);
  cudaFree(K);
}
