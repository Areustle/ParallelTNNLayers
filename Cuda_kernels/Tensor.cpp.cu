#include "Tensor.h"

#include "CudaAllocator.h"


Tensor::Tensor(size_t N, size_t C, size_t H, size_t W)
    : N(N)
    , C(C)
    , H(H)
    , W(W)
    , len(N * C * H * W) {
  cudaMalloc(&data, len * sizeof(float));
  cudaMemset(&data, 0, len);
  cudaDeviceSynchronize();
}

Tensor::Tensor(Tensor const& other)
    : N(other.N)
    , C(other.C)
    , H(other.H)
    , W(other.W)
    , len(other.len) {
  cudaMalloc(&data, len * sizeof(float));
  cudaMemcpy(data, other.data, len, cudaMemcpyDeviceToDevice);
  cudaDeviceSynchronize();
}

Tensor& Tensor::operator=(Tensor const& other) {
  if (this == &other)
    return *this;
  if (len != other.len) {
    delete[] data;
    len = other.len;
    cudaMalloc(&data, len * sizeof(float));
  }
  cudaMemcpy(data, other.data, len, cudaMemcpyDeviceToDevice);
  cudaDeviceSynchronize();
  return *this;
}

Tensor::~Tensor() {
  cudaDeviceSynchronize();
  cudaFree(data);
}
