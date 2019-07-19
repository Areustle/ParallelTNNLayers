#include "Tensor.cuh"

#include <cstring>
#include <numeric>

using namespace std;

Tensor::Tensor(std::initializer_list<int> l) : shape(l) {
  const int len =
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
  cudaMallocManaged(&m_data, len * sizeof(float));
  /* memset(m_data, 0, len); */
  for (int i = 0; i < len; ++i)
    m_data[i] = 0;
  cudaDeviceSynchronize();
}

Tensor::Tensor(Tensor const &other) : shape(other.shape) {
  cudaMallocManaged(&m_data, other.size() * sizeof(float));
  for (int i = 0; i < other.size(); ++i)
    m_data[i] = other.m_data[i];
  cudaDeviceSynchronize();
}

Tensor &Tensor::operator=(const Tensor &other) {
  if (this == &other)
    return *this;
  if (this->size() != other.size()) {
    delete[] m_data;
    cudaMallocManaged(&m_data, other.size() * sizeof(float));
  }
  for (int i = 0; i < other.size(); ++i)
    m_data[i] = other.m_data[i];
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
