#include "Utils.cuh"

#include <random>

using namespace std;


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


Tensor random_fill(std::initializer_list<unsigned> lst, float lo, float hi) {

  random_device               rd;
  mt19937                     gen(rd());
  uniform_real_distribution<> dis(lo, hi);

  Tensor A(lst);

  for (size_t i = 0; i < A.size(); ++i) A.m_data[i] = dis(gen);

  /* curandGenerator_t gen; */
  /* Tensor A(lst); */
  /* ErrChk(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT)); */
  /* ErrChk(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL)); */
  /* ErrChk(curandGenerateUniform(gen, A.m_data, A.size())); */
  /* ErrChk(curandDestroyGenerator(gen)); */

  return A;
};


__global__ void cp_recompose(float* __restrict__ Out,
                             const float* __restrict__ FilterK,
                             const float* __restrict__ FilterC,
                             const float* __restrict__ FilterH,
                             const float* __restrict__ FilterW,
                             const unsigned rank,
                             const unsigned FK,
                             const unsigned FC,
                             const unsigned FH,
                             const unsigned FW) {

  // clang-format off
  for (unsigned a = 0; a < FK; ++a)
  for (unsigned b = 0; b < FC; ++b)
  for (unsigned c = 0; c < FH; ++c)
  for (unsigned d = 0; d < FW; ++d)
  for (unsigned r = 0; r < rank; ++r)
    Out[a*FC*FH*FW + b*FH*FW + c*FW + d]
      += FilterK[a*rank + r]
       * FilterC[b*rank + r]
       * FilterH[c*rank + r]
       * FilterW[d*rank + r];
  // clang-format on
}

Tensor
cp4recom(Tensor FilterK, Tensor FilterC, Tensor FilterH, Tensor FilterW) {
  const unsigned rank = FilterK.shape[1];
  const unsigned FK   = FilterK.shape[0];
  const unsigned FC   = FilterC.shape[0];
  const unsigned FH   = FilterH.shape[0];
  const unsigned FW   = FilterW.shape[0];
  Tensor         Out  = { FK, FC, FH, FW };

  /* const unsigned WgrdDim = (W / Bw) + ((W % Bw) != 0); */
  /* const unsigned HgrdDim = (H / Bh) + ((H % Bh) != 0); */
  /* const dim3     Gshp(WgrdDim, HgrdDim, N); */
  /* const dim3     Bshp(Bw, Bh, 1); */
  /* const dim3 BlockSize = A.size() <= 512 ? A.size() : 512; */
  /* const dim3 GridSize  = (A.size() / 512) + (A.size() % 512); */

  cp_recompose<<<1, 1>>>(Out.m_data,
                         FilterK.m_data,
                         FilterC.m_data,
                         FilterH.m_data,
                         FilterW.m_data,
                         rank,
                         FK,
                         FC,
                         FH,
                         FW);

  ErrChk(cudaPeekAtLastError());
  ErrChk(cudaDeviceSynchronize());

  return Out;
}

/* template <unsigned BlockSize> */
/* __device__ void warpMax(volatile float* sdata, unsigned tid) { */
/*   if (BlockSize >= 64) sdata[tid] = fmaxf(sdata[tid], sdata[tid + 32]); */
/*   if (BlockSize >= 32) sdata[tid] = fmaxf(sdata[tid], sdata[tid + 16]); */
/*   if (BlockSize >= 16) sdata[tid] = fmaxf(sdata[tid], sdata[tid + 8]); */
/*   if (BlockSize >= 8) sdata[tid]  = fmaxf(sdata[tid], sdata[tid + 4]); */
/*   if (BlockSize >= 4) sdata[tid]  = fmaxf(sdata[tid], sdata[tid + 2]); */
/*   if (BlockSize >= 2) sdata[tid]  = fmaxf(sdata[tid], sdata[tid + 1]); */
/* } */

/* template <unsigned BlockSize> */
/* __global__ void    maxCuda(bool* __restrict__ Over, */
/*                         const float* __restrict__ delta, */
/*                         const float    tol, */
/*                         const unsigned n) { */
/*   extern __shared__ float sdata[]; */

/*   unsigned tid    = threadIdx.x; */
/*   unsigned stride = 2 * BlockSize * gridDim.x; */
/*   sdata[tid]      = 0; */

/*   for (unsigned i = blockIdx.x * (2 * BlockSize) + tid; i < n; i += stride)
 */
/*     sdata[tid]    = fmaxf(delta[i], delta[i + BlockSize]); */
/*   __syncthreads(); */

/*   if (BlockSize >= 512) { */
/*     if (tid < 256) { sdata[tid] = fmaxf(sdata[tid], sdata[tid + 256]); } */
/*     __syncthreads(); */
/*   } */
/*   if (BlockSize >= 256) { */
/*     if (tid < 128) { sdata[tid] = fmaxf(sdata[tid], sdata[tid + 128]); } */
/*     __syncthreads(); */
/*   } */
/*   if (BlockSize >= 128) { */
/*     if (tid < 64) { sdata[tid] = fmaxf(sdata[tid], sdata[tid + 64]); } */
/*     __syncthreads(); */
/*   } */

/*   if (tid < 32) warpMax(sdata, tid); */
/*   if (tid == 0) Over[blockIdx.x] = sdata[0]; */
/* } */

__global__ void arrayRelativeDelta(int*        Exceeds,
                                   const float tol,
                                   const float* __restrict__ A,
                                   const float* __restrict__ B) {
  unsigned tid = threadIdx.x + blockIdx.x * blockDim.x;
  float    a   = A[tid];
  float    b   = B[tid];
  float    d   = fabsf(a - b);
  float    err = (a == 0 || b == 0) ? d : fmaxf(d / fabsf(a), d / fabsf(b));
  if (err > tol) atomicCAS(Exceeds, 1, 0);
}

bool AllClose(Tensor A, Tensor B, float tolerance) {
  if (A.size() != B.size()) return false;

  int* Exceeds;
  cudaMalloc(&Exceeds, sizeof(int));
  cudaMemset(&Exceeds, 1, 1);

  unsigned BlockSize = A.size() <= 512 ? A.size() : 512;
  unsigned GridSize  = (A.size() / 512) + (A.size() % 512);
  arrayRelativeDelta<<<BlockSize, GridSize>>>(
      Exceeds, tolerance, A.m_data, B.m_data);
  ErrChk(cudaPeekAtLastError());
  ErrChk(cudaDeviceSynchronize());

  int host_Exceeds;
  cudaMemcpy(&host_Exceeds, Exceeds, 1, cudaMemcpyDeviceToHost);

  return host_Exceeds;
}
