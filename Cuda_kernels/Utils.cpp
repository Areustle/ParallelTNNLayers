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
cp4recom(Tensor FilterK, Tensor FilterC, Tensor FilterR, Tensor FilterS) {
  const size_t rank = FilterK.shape[1];
  const int    FK   = FilterK.shape[0];
  const int    FC   = FilterC.shape[0];
  const int    FR   = FilterR.shape[0];
  const int    FS   = FilterS.shape[0];
  Tensor       Out  = { FK, FC, FR, FS };

  // clang-format off
  for (int a = 0; a < FK; ++a)
  for (int b = 0; b < FC; ++b)
  for (int c = 0; c < FR; ++c)
  for (int d = 0; d < FS; ++d)
  for (int r = 0; r < rank; ++r)
    Out[a*FC*FR*FS + b*FR*FS + c*FS + d]
      += FilterK[a*rank + r]
       * FilterC[b*rank + r]
       * FilterR[c*rank + r]
       * FilterS[d*rank + r];
  // clang-format on

  return Out;
}
