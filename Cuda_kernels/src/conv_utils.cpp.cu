#include "cpu_conv2d.h"
#include "Tensor.h"

#include <random>

using namespace std;

Tensor random_fill(size_t N, size_t C, size_t H, size_t W, float lo, float hi) {

  random_device               rd;
  mt19937                     gen(rd());
  uniform_real_distribution<> dis(lo, hi);

  Tensor A(N, C, H, W);

  for (size_t i = 0; i < A.size(); ++i)
    A[i] = dis(gen);

  return A;
};
