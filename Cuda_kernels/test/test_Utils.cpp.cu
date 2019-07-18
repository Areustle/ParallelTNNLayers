#define DOCTEST_CONFIG_IMPLEMENTATION_IN_DLL
#include "../../external/doctest/doctest.h"
#include "../Tensor.h"
#include "../Utils.h"
#include "../cudnnConv2d.h"
#include "../manual.h"

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
}

TEST_CASE("Padding test") {

  Tensor U = random_fill({ 3, 4, 5, 6 }, 0, 1);
  REQUIRE(U.shape[0] == 3);
  REQUIRE(U.shape[1] == 4);
  REQUIRE(U.shape[2] == 5);
  REQUIRE(U.shape[3] == 6);

  Tensor padU = padNCHW(U, 1);
  REQUIRE(padU.shape[0] == 3);
  REQUIRE(padU.shape[1] == 4);
  REQUIRE(padU.shape[2] == 7);
  REQUIRE(padU.shape[3] == 8);

  int N  = U.shape[0];
  int C  = U.shape[1];
  int H  = U.shape[2];
  int W  = U.shape[3];
  int oH = padU.shape[2];
  int oW = padU.shape[3];


  // clang-format off
  for (int n = 0; n < N; ++n)
  for (int c = 0; c < C; ++c)
  for (int h = 0; h < oH; ++h)
    for (int w = 0; w < oW; ++w) {
      int i = n*C*H*W + c*H*W + (h-1)*W + (w-1);
      int p = n*C*oH*oW + c*oH*oW + h*oW + w;
      if (h >= 1 && h < (H+1) && w >= 1 && w < (W+1)) {
        REQUIRE(padU[p] == doctest::Approx(U[i]));
      } else {
        REQUIRE(padU[p] == 0);
      }
    }
}
