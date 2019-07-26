#define DOCTEST_CONFIG_IMPLEMENTATION_IN_DLL
#include "../../external/doctest/doctest.h"
#include "../Tensor.cuh"
#include "../Utils.cuh"
#include "../cp4Conv2d.cuh"
/* #include "../conv.cuh" */
#include "../cudnnConv2d.cuh"
#include "../manual.cuh"

TEST_CASE("Convolution test") {

  int  x     = 32;
  int  c     = 16;
  int  k     = 16;
  int  n     = 4;
  int  rank  = 6;
  auto U     = random_fill({ n, c, x, x }, 0, 1);
  auto K0    = random_fill({ k, rank }, 0, 1);
  auto K1    = random_fill({ c, rank }, 0, 1);
  auto K2    = random_fill({ 3, rank }, 0, 1);
  auto K3    = random_fill({ 3, rank }, 0, 1);
  auto K     = cp4recom(K0, K1, K2, K3);
  auto Cudnn = nn_conv2d(U, K);

  auto padU = padNCHW(U, 1);
  auto Full_gpu = conv2d_cp4_gpu(padU, K0, K1, K2, K3, 1);

  REQUIRE(Cudnn.size() == Full_gpu.size());
  REQUIRE(Full_gpu.shape[0] == n);
  REQUIRE(Full_gpu.shape[1] == k);
  REQUIRE(Full_gpu.shape[2] == x);
  REQUIRE(Full_gpu.shape[3] == x);
  for (int i = 0; i < Cudnn.size(); ++i)
    REQUIRE(Cudnn.m_data[i]
            == doctest::Approx(Full_gpu.m_data[i]).epsilon(1e-5));
}
