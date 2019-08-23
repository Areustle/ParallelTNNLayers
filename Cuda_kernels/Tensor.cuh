#ifndef TENSOR_H
#define TENSOR_H

#include <cstddef>
#include <initializer_list>
#include <memory>
#include <tuple>
#include <vector>

struct tensor_shape {
  unsigned N     = 0;
  unsigned C     = 0;
  unsigned H     = 0;
  unsigned W     = 0;
  unsigned pad   = 0;
  unsigned fK    = 0;
  unsigned fC    = 0;
  unsigned fH    = 0;
  unsigned fW    = 0;
  unsigned fRank = 0;

  friend bool operator<(const tensor_shape l, const tensor_shape r) {
    return std::tie(l.N, l.C, l.H, l.W, l.pad, l.fK, l.fC, l.fH, l.fW, l.fRank)
           < std::tie(
               r.N, r.C, r.H, r.W, r.pad, r.fK, l.fC, r.fH, r.fW, r.fRank);
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
