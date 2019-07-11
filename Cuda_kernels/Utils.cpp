#include "Utils.h"

#include "Tensor.h"
#include <random>

using namespace std;

Tensor random_fill(std::initializer_list<int> lst, float lo, float hi) {

  random_device               rd;
  mt19937                     gen(rd());
  uniform_real_distribution<> dis(lo, hi);

  Tensor A(lst);

  for (size_t i = 0; i < A.size(); ++i) A[i] = dis(gen);

  return A;
};

Tensor
cp4recom(Tensor Kernel0, Tensor Kernel1, Tensor Kernel2, Tensor Kernel3) {
  const size_t rank = Kernel0.shape[1];
  const int    K0   = Kernel0.shape[0];
  const int    K1   = Kernel1.shape[0];
  const int    K2   = Kernel2.shape[0];
  const int    K3   = Kernel3.shape[0];
  Tensor       Out  = { K0, K1, K2, K3 };

  /* // clang-format off */
  for (int r = 0; r < rank; ++r) {
    for (int a = 0; a < K0; ++a)
    for (int b = 0; b < K1; ++b)
    for (int c = 0; c < K2; ++c)
    for (int d = 0; d < K3; ++d)
      Out[a*K1*K2*K3 + b*K2*K3 + c*K3 + d]
        += Kernel0[a*rank + r]
         * Kernel1[b*rank + r]
         * Kernel2[c*rank + r]
         * Kernel3[d*rank + r];
  }
  // clang-format on

  return Out;
}
