

__constant__ float const_filter[4096];

template<unsigned TileFactor = 1>
__global__ void conv2d_cp4_kernel(const float* __restrict__ Input,
                                  const unsigned pad,
                                  const unsigned offset_fK,
                                  const unsigned offset_fC,
                                  const unsigned offset_fH,
                                  const unsigned offset_fW,
                                  const unsigned Rank,
                                  const unsigned fK,
                                  const unsigned fC,
                                  const unsigned fH,
                                  const unsigned fW,
                                  const unsigned N,
                                  const unsigned C,
                                  float* __restrict__ Out) {

  extern __shared__ float shared_mem[];

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

  /* float local[32]; */

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
  for (unsigned fh = 0; fh < fH; ++fh)
  for (unsigned fw = 0; fw < fW; ++fw)
  for (unsigned rr = 0; rr < Rank; ++rr)
  for (unsigned t = 0; t < TileFactor; ++t)
    sum[t] += shared_mem[c*sH*sW + (h+fh)*sW + (w+fw+(t*Bw))]
           *  const_filter[offset_fK + k*Rank + rr]
           *  const_filter[offset_fC + c*Rank + rr]
           *  const_filter[offset_fH + fh*Rank + rr]
           *  const_filter[offset_fW + fw*Rank + rr];

  // populate output array.
  #pragma unroll
  for (unsigned t = 0; t < TileFactor; ++t)
    Out[h*oW + w+(t*Bw)] = sum[t];

  // clang-format on
}


int main() {

  float* In;
  float* Out;
  void*  FilterK;
  void*  FilterC;
  void*  FilterW;
  void*  FilterH;

  unsigned N    = 1;
  unsigned C    = 16;
  unsigned H    = 32;
  unsigned W    = 32;
  unsigned fK   = 16;
  unsigned fH   = 3;
  unsigned fW   = 3;
  unsigned Rank = 1;


  cudaMalloc(&In, (N * C * (H + 2) * (W + 2)) * sizeof(float));
  cudaMalloc(&FilterK, (fK * Rank) * sizeof(float));
  cudaMalloc(&FilterC, (C * Rank) * sizeof(float));
  cudaMalloc(&FilterH, (fH * Rank) * sizeof(float));
  cudaMalloc(&FilterW, (fW * Rank) * sizeof(float));
  cudaMalloc(&Out, (N * fK * H * W) * sizeof(float));

  static const unsigned tf   = 2;
  const unsigned        bdim = 16;
  // Populate GPU constant memory with the 4 filters at an appropriate offset.
  const size_t shim_sz   = C * (fW - 1 + bdim * tf) * (fH - 1 + bdim);
  const size_t offset_fK = shim_sz; // 0
  const size_t offset_fC = offset_fK + (fK * Rank);
  const size_t offset_fH = offset_fC + (C * Rank);
  const size_t offset_fW = offset_fH + (fH * Rank);
  cudaMemcpyToSymbol(const_filter,
                     FilterK,
                     sizeof(float) * (fK * Rank),
                     sizeof(float) * offset_fK);
  cudaMemcpyToSymbol(const_filter,
                     FilterC,
                     sizeof(float) * (C * Rank),
                     sizeof(float) * offset_fC);
  cudaMemcpyToSymbol(const_filter,
                     FilterH,
                     sizeof(float) * fH * Rank,
                     sizeof(float) * offset_fH);
  cudaMemcpyToSymbol(const_filter,
                     FilterW,
                     sizeof(float) * fW * Rank,
                     sizeof(float) * offset_fW);

  const size_t smsz =
      (shim_sz + (fK * Rank) + (C * Rank) + (fH * Rank) + (fW * Rank)) * //
      sizeof(float);

  const dim3 Gshp(W / (bdim * tf), H / (bdim), fK * N);
  const dim3 Bshp(bdim, bdim, 1);
  conv2d_cp4_kernel<tf><<<Gshp, Bshp, smsz>>>(In,        //
                                              2,         //
                                              offset_fK, //
                                              offset_fC, //
                                              offset_fH, //
                                              offset_fW, //
                                              Rank,
                                              fK, //
                                              C,  //
                                              fH, //
                                              fW, //
                                              N,  //
                                              C,  //
                                              Out);
  cudaDeviceSynchronize();

  cudaFree(In);
  cudaFree(FilterK);
  cudaFree(FilterC);
  cudaFree(FilterH);
  cudaFree(FilterW);
  cudaFree(Out);
}
