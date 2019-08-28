#include "Utils.cuh"
#include "iostream"

#include <curand.h>

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


Tensor random_fill(std::initializer_list<unsigned> lst) {

  curandGenerator_t gen;
  Tensor            A(lst);
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
  curandGenerateUniform(gen, A.m_data, A.size());
  cudaDeviceSynchronize();
  curandDestroyGenerator(gen);

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
  for (unsigned fc = threadIdx.z + blockIdx.z * blockDim.z; fc < FC; fc += blockDim.z*gridDim.z)
  for (unsigned fh = threadIdx.y + blockIdx.y * blockDim.y; fh < FH; fh += blockDim.y*gridDim.y)
  for (unsigned fw = threadIdx.x + blockIdx.x * blockDim.x; fw < FW; fw += blockDim.x*gridDim.x)
  for (unsigned fk = 0; fk < FK; ++fk) {

    float pixel = 0.0f;

    for (unsigned rr = 0; rr < rank; ++rr) {
      pixel += FilterK[fk * rank + rr]
             * FilterC[fc * rank + rr]
             * FilterH[fh * rank + rr]
             * FilterW[fw * rank + rr];
    }

    Out[fk*FC*FH*FW + fc*FH*FW + fh*FW + fw] = pixel;
  }
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

  const unsigned W_dim = (FW / 8) + ((FW % 8) != 0);
  const unsigned H_dim = (FH / 8) + ((FH % 8) != 0);
  const unsigned C_dim = (FC / 8) + ((FC % 8) != 0);
  const dim3     Gshp(W_dim, H_dim, C_dim);
  const dim3     Bshp(8, 8, 8);

  cp_recompose<<<Gshp, Bshp>>>(Out.m_data,
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
                                   const int   size,
                                   const float* __restrict__ A,
                                   const float* __restrict__ B) {
  unsigned tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= size) return;

  float a = A[tid];
  float b = B[tid];
  float d = fabsf(a - b);
  float err = (a == 0 || b == 0) ? d : fmaxf(d / fabsf(a), d / fabsf(b));
  if (err > tol) atomicAdd(Exceeds, 1);
}


bool AllClose(Tensor A, Tensor B, float tolerance) {
  if (A.size() != B.size()) { return false; }

  int* Exceeds;
  ErrChk(cudaMalloc(&Exceeds, sizeof(int)));
  ErrChk(cudaMemset(Exceeds, 0, sizeof(int)));

  unsigned BlockSize = A.size() < 512 ? A.size() : 512;
  unsigned GridSize = (A.size() / 512) + ((A.size() % 512) != 0);

  arrayRelativeDelta<<<GridSize, BlockSize>>>(
      Exceeds, tolerance, A.size(), A.m_data, B.m_data);

  ErrChk(cudaPeekAtLastError());
  ErrChk(cudaDeviceSynchronize());

  int host_Exceeds;
  ErrChk(
      cudaMemcpy(&host_Exceeds, Exceeds, sizeof(int), cudaMemcpyDeviceToHost));

  return host_Exceeds == 0;
}
