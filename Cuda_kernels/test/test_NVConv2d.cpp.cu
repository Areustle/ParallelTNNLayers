#define DOCTEST_CONFIG_IMPLEMENTATION_IN_DLL
#include "../../external/doctest/doctest.h"
#include "../NVConv2d.cuh"
#include <random>

TEST_CASE("cudnn_full_conv2d test") {
  Tensor K = { 1, 4, 3, 3 };
  for (int i = 0; i < K.size(); ++i) REQUIRE(K.m_data[i] == 0);

  Tensor U = { 1, 4, 32, 32 };

  std::random_device               rd;
  std::mt19937                     gen(rd());
  std::uniform_real_distribution<> dis(0.1, 1.0);

  for (size_t i = 0; i < U.size(); ++i) U.m_data[i] = dis(gen);
  for (int i = 0; i < U.size(); ++i) REQUIRE(U.m_data[i] != 0);
  for (int i = 0; i < K.size(); ++i) REQUIRE(K.m_data[i] == 0);

  auto V = NV::Conv2dForward(U, K, 1);

  for (int i = 0; i < V.size(); ++i) REQUIRE(V.m_data[i] == 0);
}
