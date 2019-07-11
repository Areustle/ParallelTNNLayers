#include "Tensor.h"

#include <numeric>

using namespace std;

Tensor::Tensor(std::initializer_list<int> l)
    : shape(l) {
  const int len =
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
  cudaMallocManaged(&m_data, len * sizeof(float));
  cudaMemset(&m_data, 0, len);
  cudaDeviceSynchronize();
}


Tensor::Tensor(Tensor const& other)
    : shape(other.shape) {
  const int len =
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
  cudaMallocManaged(&m_data, len * sizeof(float));
  cudaMemcpy(
      m_data, other.m_data, len * sizeof(float), cudaMemcpyDeviceToDevice);
  cudaDeviceSynchronize();
}


Tensor& Tensor::operator=(const Tensor& other) {
  if (this == &other)
    return *this;
  if (this->size() != other.size()) {
    delete[] m_data;
    cudaMallocManaged(&m_data, other.size() * sizeof(float));
  }
  cudaMemcpy(m_data,
             other.m_data,
             other.size() * sizeof(float),
             cudaMemcpyDeviceToDevice);
  cudaDeviceSynchronize();
  return *this;
}


Tensor::~Tensor() {
  cudaDeviceSynchronize();
  cudaFree(m_data);
}

size_t Tensor::size() const {
  return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
}
