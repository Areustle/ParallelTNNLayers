#include "Tensor.h"

Tensor::Tensor(size_t N, size_t C, size_t H, size_t W)
    : N(N)
    , C(C)
    , H(H)
    , W(W) {

  size_t len = N * C * H * W;
  data       = new float[len];

  for (size_t i = 0; i < len; ++i)
    data[i] = 0;
}

Tensor::~Tensor() { delete[] data; }
