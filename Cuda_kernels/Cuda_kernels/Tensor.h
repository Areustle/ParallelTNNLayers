#ifndef TENSOR_H
#define TENSOR_H

#include <cstddef>
#include <initializer_list>
#include <memory>
#include <vector>


class Tensor
{
public:
  float*                 m_data;
  std::vector<int const> shape;

  Tensor(std::initializer_list<int const>);
  Tensor(Tensor const&);
  Tensor(Tensor&&) = default;
  Tensor& operator =(Tensor const&);
  Tensor& operator=(Tensor&&) = default;
  ~Tensor();

  float* get() { return m_data; }
  size_t size(); // { return shape.size(); }
  size_t rank() { return shape.size(); }

  float& operator[](size_t const index) { return m_data[index]; }
};

#endif /* TENSOR_H */
