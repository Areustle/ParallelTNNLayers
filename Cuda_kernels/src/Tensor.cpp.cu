#include "Tensor.h"

using namespace std;

Tensor::Tensor(size_t N, size_t C, size_t H, size_t W)
    : N(N)
    , C(C)
    , H(H)
    , W(W)
    , len(N * C * H * W) {
  cudaMallocManaged(&m_data, len * sizeof(float));
  cudaMemset(&m_data, 0, len);
  cudaDeviceSynchronize();
}

Tensor::Tensor(Tensor const& other)
    : N(other.N)
    , C(other.C)
    , H(other.H)
    , W(other.W)
    , len(other.len) {
  cudaMallocManaged(&m_data, len * sizeof(float));
  cudaMemcpy(
      m_data, other.m_data, len * sizeof(float), cudaMemcpyDeviceToDevice);
  cudaDeviceSynchronize();
}

Tensor& Tensor::operator=(Tensor const& other) {
  if (this == &other)
    return *this;
  if (len != other.len) {
    delete[] m_data;
    len = other.len;
    cudaMallocManaged(&m_data, len * sizeof(float));
  }
  cudaMemcpy(
      m_data, other.m_data, len * sizeof(float), cudaMemcpyDeviceToDevice);
  cudaDeviceSynchronize();
  return *this;
}

Tensor::~Tensor() {
  cudaDeviceSynchronize();
  cudaFree(m_data);
}
