#include "conv_utils.h"

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

  /* Tensor cp4recom(Tensor A, Tensor B, Tensor C, Tensor D) { */

  /*   size_t rank = A.C; */
  /*   Tensor Out(A.W, B.W, C.W, D.W); */

  /*   for (int r = 0; r < rank; ++r) { */
  /*     size_t rr = r * rank; */
  /*     for (int a = 0; a < A.W; ++a) */
  /*       for (int b = 0; b < B.W; ++b) */
  /*         for (int c = 0; c < C.W; ++c) */
  /*           for (int d = 0; d < D.W; ++d) */
  /*             Out[a * A.W * B.W * C.W + b * B.W * C.W + c * C.W + d] += */
  /*                 A[rr + a] * B[rr + b] * C[rr + c] * D[rr + d]; */
  /*   } */

  /*   return Out; */
  /* } */

#define DOCTEST_CONFIG_IMPLEMENTATION_IN_DLL
#include "cudnn_full_conv2d.h"
#include "../external/doctest/doctest.h"
#include "manual.h"

TEST_CASE("Conv utils test") {

  Tensor K0{ 6, 16, 1, 1 };
  Tensor K1{ 6, 1, 3, 1 };
  Tensor K2{ 6, 1, 1, 3 };
  Tensor K3{ 16, 6, 1, 1 };
  Tensor U = random_fill({ 1, 16, 32, 32 }, 0, 1);

  for (int i = 0; i < K0.size(); ++i) REQUIRE(K0[i] == 0);
  for (int i = 0; i < K1.size(); ++i) REQUIRE(K1[i] == 0);
  for (int i = 0; i < K2.size(); ++i) REQUIRE(K2[i] == 0);
  for (int i = 0; i < K3.size(); ++i) REQUIRE(K3[i] == 0);

  /* Tensor K = cp4recom(K0, K1, K2, K3); */
  /* CHECK(K.size() == 2304); */

  for (int i = 0; i < U.size(); ++i) REQUIRE(U[i] > 0);

  auto U0 = nn_conv2d(U, K0);
  for (int i = 0; i < U0.size(); ++i) REQUIRE(U0[i] == 0);
  /* Tensor U1 = nn_conv2d(U0, K1); */
  /* Tensor U2 = nn_conv2d(U1, K2); */
  /* Tensor V1 = nn_conv2d(U2, K3); */
  /* Tensor V2 = static_cp4_conv2d(U, K0, K1, K2, K3); */

  /* REQUIRE(V1.size() == V2.size()); */

  /* for (int i = 0; i < V1.size(); ++i) { */
  /*   REQUIRE(V1[i] == V2[i]); */
  /* } */
}
