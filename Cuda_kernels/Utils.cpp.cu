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

  /*
     2 x 5 (Rows, Cols)

     01234
     56789

     [r*Cols + c]
  */

  // clang-format off
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

#define DOCTEST_CONFIG_IMPLEMENTATION_IN_DLL
#include "../external/doctest/doctest.h"
#include "cudnnConv2d.h"
#include "manual.h"

TEST_CASE("Utils test") {

  Tensor K0 = random_fill({ 16, 6 });
  Tensor K1 = random_fill({ 16, 6 });
  Tensor K2 = random_fill({ 3, 6 });
  Tensor K3 = random_fill({ 3, 6 });
  Tensor U  = random_fill({ 1, 16, 32, 32 }, 0, 1);
  for (int i = 0; i < U.size(); ++i) REQUIRE(U[i] > 0);

  Tensor K = cp4recom(K0, K1, K2, K3);
  CHECK(K.size() == 2304);
  CHECK(K.order() == 4);
  CHECK(K.shape[0] == 16);
  CHECK(K.shape[1] == 16);
  CHECK(K.shape[2] == 3);
  CHECK(K.shape[3] == 3);

  auto V = nn_conv2d(U, K);
  CHECK(V.size() == (1 * 16 * 32 * 32));
  CHECK(V.order() == 4);
  CHECK(V.shape[0] == 1);
  CHECK(V.shape[1] == 16);
  CHECK(V.shape[2] == 32);
  CHECK(V.shape[3] == 32);
  for (int i = 0; i < V.size(); ++i) REQUIRE(V[i] != 0);
  /* Tensor W = static_cp4_conv2d(U, K0, K1, K2, K3); */

  /* REQUIRE(V1.size() == V2.size()); */

  /* for (int i = 0; i < V1.size(); ++i) { */
  /*   REQUIRE(V1[i] == V2[i]); */
  /* } */
}
