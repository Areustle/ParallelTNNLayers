#include "Tensor.h"

#include "CudaAllocator.h"


template<typename T, class A>
Tensor<T, A>::Tensor(size_t N, size_t C, size_t H, size_t W)
    : N(N)
    , C(C)
    , H(H)
    , W(W)
    , len(N * C * H * W) {
  T* buf = std::allocator_traits<A>::allocate(Alloc, len);
  data   = new (buf) T(0);
}

template<typename T, class A>
Tensor<T, A>::Tensor(Tensor<T, A> const& other)
    : N(other.N)
    , C(other.C)
    , H(other.H)
    , W(other.W)
    , len(N * C * H * W) {
  T* buf = std::allocator_traits<A>::allocate(Alloc, len);
  data   = new (buf) T(*other.data);
}

template<typename T, class A>
Tensor<T, A>& Tensor<T, A>::operator=(Tensor<T, A> const& other) {
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


template class Tensor<float, std::allocator<float>>;
template class Tensor<float, CudaAllocator>;
