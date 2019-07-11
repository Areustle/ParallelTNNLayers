#define DOCTEST_CONFIG_IMPLEMENTATION_IN_DLL
#include "../cudnnConv2d.h"
#include "../../external/doctest/doctest.h"
#include <random>

TEST_CASE("cudnn_full_conv2d test") {
  Tensor U{ 1, 4, 32, 32 };
  Tensor K{ 1, 4, 3, 3 };

  std::random_device               rd;
  std::mt19937                     gen(rd());
  std::uniform_real_distribution<> dis(0.1, 1.0);

  for (size_t i = 0; i < U.size(); ++i) U[i] = dis(gen);
  for (int i = 0; i < U.size(); ++i) CHECK(U[i] != 0);
  for (int i = 0; i < K.size(); ++i) CHECK(K[i] == 0);

  auto V = nn_conv2d(U, K);

  for (int i = 0; i < V.size(); ++i) CHECK(V[i] == 0);
}
