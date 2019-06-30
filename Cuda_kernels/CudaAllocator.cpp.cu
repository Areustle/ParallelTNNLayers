#include "CudaAllocator.h"

float* CudaAllocator::allocate(std::size_t n) {
  float* p;
  cudaMallocManaged(&p, n * sizeof(float));
  cudaDeviceSynchronize();
  return p;
}

void CudaAllocator::deallocate(float* p, std::size_t n) {
  cudaDeviceSynchronize();
  cudaFree(p);
}


