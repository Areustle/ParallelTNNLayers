#include "Tensor.h"

#include "CudaAllocator.h"


template<class A>
Tensor<A>::Tensor(size_t N, size_t C, size_t H, size_t W)
    : N(N)
    , C(C)
    , H(H)
    , W(W)
    , len(N * C * H * W) {
  float* buf = std::allocator_traits<A>::allocate(Alloc, len);
  data       = new (buf) float(0);
}

template<class A>
Tensor<A>::Tensor(Tensor<A> const& other)
    : N(other.N)
    , C(other.C)
    , H(other.H)
    , W(other.W)
    , len(N * C * H * W) {
  float* buf = std::allocator_traits<A>::allocate(Alloc, len);
  data       = new (buf) float(*other.data);
}

template<class A> Tensor<A>& Tensor<A>::operator=(Tensor<A> const& other) {
  if (this == &other)
    return *this;
  if (len != other.len) {
    delete[] data;
    len  = other.len;
    data = std::allocator_traits<A>::allocate(Alloc, len);
  }
  std::copy(&other.data[0], &other.data[0] + other.len, &data[0]);
  return *this;
}

template<class A> Tensor<A>::~Tensor() {
  std::allocator_traits<A>::deallocate(Alloc, data, len);
}

template class Tensor<CudaAllocator>;
template class Tensor<std::allocator<float>>;
