#ifndef TENSOR_H
#define TENSOR_H

#include "CudaAllocator.h"
#include <cstddef>
#include <memory>

class Tensor
{
public:
  size_t N;
  size_t C;
  size_t H;
  size_t W;
  size_t size;
  float* m_data;

  Tensor(size_t N, size_t C, size_t H, size_t W);

  Tensor(Tensor const& other);

  Tensor& operator=(Tensor const& other);

  ~Tensor();

  float* get() { return m_data; }
  size_t len() { return size; }

  float& operator[](size_t const index) { return m_data[index]; }
};

#endif /* TENSOR_H */