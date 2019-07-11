#define DOCTEST_CONFIG_IMPLEMENTATION_IN_DLL
#include "../../external/doctest/doctest.h"
#include "../Tensor.h"
#include "../manual.h"
#include "../Utils.h"
#include "../cudnnConv2d.h"

TEST_CASE("Utils test") {

  auto K0 = random_fill({ 16, 6 }, 0, 1);
  auto K1 = random_fill({ 16, 6 }, 0, 1);
  auto K2 = random_fill({ 3, 6 }, 0, 1);
  auto K3 = random_fill({ 3, 6 }, 0, 1);
  auto U  = random_fill({ 1, 16, 32, 32 }, 0, 1);
  auto K  = cp4recom(K0, K1, K2, K3);
  auto V  = nn_conv2d(U, K);
  auto W  = static_cp4_conv2d(U, K1, K2, K3, K0);

  REQUIRE(V.size() == W.size());

  /* REQUIRE(V[0] == W[0]); */
  REQUIRE(V[0] == doctest::Approx(W[0]).epsilon(1e-3));
  for (int i = 0; i < V.size(); ++i)
    REQUIRE(V[i] == doctest::Approx(W[i]).epsilon(1e-3));
}
