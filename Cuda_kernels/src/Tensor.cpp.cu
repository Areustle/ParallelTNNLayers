#include "Tensor.h"

#include "CudaAllocator.h"


Tensor::Tensor(size_t N, size_t C, size_t H, size_t W)
    : N(N)
    , C(C)
    , H(H)
    , W(W)
    , size(N * C * H * W) {
  cudaMallocManaged(&m_data, size * sizeof(float));
  cudaMemset(&m_data, 0, size);
  cudaDeviceSynchronize();
}

Tensor::Tensor(Tensor const& other)
    : N(other.N)
    , C(other.C)
    , H(other.H)
    , W(other.W)
    , size(other.size) {
  cudaMallocManaged(&m_data, size * sizeof(float));
  cudaMemcpy(m_data, other.m_data, size * sizeof(float), cudaMemcpyDeviceToDevice);
  cudaDeviceSynchronize();
}

Tensor& Tensor::operator=(Tensor const& other) {
  if (this == &other)
    return *this;
  if (size != other.size) {
    delete[] m_data;
    size = other.size;
    cudaMallocManaged(&m_data, size * sizeof(float));
  }
  cudaMemcpy(m_data, other.m_data, size * sizeof(float), cudaMemcpyDeviceToDevice);
  cudaDeviceSynchronize();
  return *this;
}

Tensor::~Tensor() {
  cudaDeviceSynchronize();
  cudaFree(m_data);
}
