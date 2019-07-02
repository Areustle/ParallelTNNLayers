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
  size_t len;
  float* data;

  Tensor(size_t N, size_t C, size_t H, size_t W);

  Tensor(Tensor const& other);

  Tensor& operator=(Tensor const& other);

  ~Tensor();

  float* get() { return data; }
  size_t size() { return len; }
};

#endif /* TENSOR_H */
