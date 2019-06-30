#ifndef TENSOR_H
#define TENSOR_H

#include <cstddef>
#include <memory>

template<typename T, class A = std::allocator<T>>
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
  T*     data;

  Tensor(size_t N, size_t C, size_t H, size_t W);

  Tensor(Tensor const& other);

  Tensor& operator=(Tensor const& other);

  ~Tensor() { std::allocator_traits<A>::deallocate(Alloc, data, len); }

  T*     get() { return data; }
  size_t size() { return len; }


  template<class U>
  Tensor(Tensor<T, U> const& other)
      : N(other.N)
      , C(other.C)
      , H(other.H)
      , W(other.W)
      , len(N * C * H * W) {
    T* buf = std::allocator_traits<A>::allocate(Alloc, len);
    data   = new (buf) T(*other.data);
  }

  template<class U>
  Tensor& operator=(Tensor<T, U> const& other) {
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
