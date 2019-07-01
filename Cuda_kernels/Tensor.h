#ifndef TENSOR_H
#define TENSOR_H

#include "CudaAllocator.h"
#include <cstddef>
#include <memory>

template<class A = CudaAllocator>
class Tensor
{
private:
  A Alloc;

public:
  size_t N;
  size_t C;
  size_t H;
  size_t W;
  size_t len;
  float*     data;

  Tensor(size_t N, size_t C, size_t H, size_t W);

  Tensor(Tensor const& other);

  Tensor& operator=(Tensor const& other);

  ~Tensor() { std::allocator_traits<A>::deallocate(Alloc, data, len); }

  float*     get() { return data; }
  size_t size() { return len; }


  template<class U>
  Tensor(Tensor<U> const& other)
      : N(other.N)
      , C(other.C)
      , H(other.H)
      , W(other.W)
      , len(N * C * H * W) {
    float* buf = std::allocator_traits<A>::allocate(Alloc, len);
    data   = new (buf) float(*other.data);
  }

  template<class U>
  Tensor& operator=(Tensor<U> const& other) {
    if (this == &other)
      return *this;
    if (len != other.len) {
      std::allocator_traits<A>::deallocate(Alloc, data, len);
      len  = other.len;
      data = std::allocator_traits<A>::allocate(Alloc, len);
    }
    std::copy(&other.data[0], &other.data[0] + other.len, &data[0]);
    return *this;
  }
};

#endif /* TENSOR_H */
