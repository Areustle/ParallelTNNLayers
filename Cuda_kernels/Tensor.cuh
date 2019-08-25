#ifndef TENSOR_H
#define TENSOR_H

#include <cstddef>
#include <initializer_list>
#include <memory>
#include <tuple>
#include <vector>

struct tensor_shape {
  unsigned N    = 0;
  unsigned H    = 0;
  unsigned W    = 0;
  unsigned pad  = 0;
  unsigned T    = 0;
  unsigned C    = 0;
  unsigned Y    = 0;
  unsigned X    = 0;
  unsigned Rank = 0;

  friend bool operator<(const tensor_shape l, const tensor_shape r) {
    return std::tie(l.N, l.H, l.W, l.pad, l.T, l.C, l.Y, l.X, l.Rank)
           < std::tie(r.N, r.H, r.W, r.pad, r.T, l.C, r.Y, r.X, r.Rank);
  }
};


class Tensor
{
public:
  float*                m_data;
  std::vector<unsigned> shape;

  Tensor(std::initializer_list<unsigned>);
  Tensor(Tensor const&);
  Tensor(Tensor&&) = default;
  Tensor& operator =(Tensor const&);
  Tensor& operator=(Tensor&&) = default;
  ~Tensor();

  size_t size() const;
  size_t order() { return shape.size(); }
  /* float& operator[](size_t const index) { return m_data[index]; } */
};

#endif /* TENSOR_H */
