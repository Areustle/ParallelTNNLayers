#define DOCTEST_CONFIG_IMPLEMENTATION_IN_DLL
#include "../../external/doctest/doctest.h"
#include "../Tensor.cuh"
#include "../Utils.cuh"
/* #include "../cp4Conv2d.h" */
#include "../conv.cuh"
#include "../cudnnConv2d.cuh"
#include "../manual.cuh"

TEST_CASE("Convolution test") {

  int  x = 32;
  int  c = 4;
  int  k = 4;
  int  n = 1;
  auto U = random_fill({ n, c, x, x }, 0, 1);
  auto K = random_fill({ k, c, 3, 3 }, 0, 1);
  /* auto K     = cp4recom(K0, K1, K2, K3); */
  auto Cudnn = nn_conv2d(U, K);
  /* auto cpu_full = conv2d_full_cpu(U, K); */
  /* for (int i = 0; i < Cudnn.size(); ++i) */
  /*   REQUIRE(Cudnn[i] == doctest::Approx(cpu_full[i]).epsilon(1e-5)); */

  auto padU = padNCHW(U, 1);

  REQUIRE(padU.size() == (n * c * (x + 2) * (x + 2)));
  REQUIRE(padU.shape[0] == n);
  REQUIRE(padU.shape[1] == c);
  REQUIRE(padU.shape[2] == x + 2);
  REQUIRE(padU.shape[3] == x + 2);
  /* for (int i = 0; i < (x + 2); ++i) REQUIRE(padU.m_data[i] == 0); */
  /* for (int i = (x + 1) * (x + 2); i < (x + 2) * (x + 2); ++i) */
  /*   REQUIRE(padU.m_data[i] == 0); */
  /* for (int i = 0; i < (x + 2) * (x + 2); i += (x + 2)) { */
  /*   REQUIRE(padU.m_data[i] == 0); */
  /*   REQUIRE(padU.m_data[i + (x + 1)] == 0); */
  /* } */

  auto Full_gpu = conv2d_full_gpu(padU, K);

  REQUIRE(Cudnn.size() == Full_gpu.size());
  REQUIRE(Full_gpu.shape[0] == n);
  REQUIRE(Full_gpu.shape[1] == k);
  REQUIRE(Full_gpu.shape[2] == x);
  REQUIRE(Full_gpu.shape[3] == x);
  for (int i = 0; i < Cudnn.size(); ++i)
    REQUIRE(Cudnn.m_data[i]
            == doctest::Approx(Full_gpu.m_data[i]).epsilon(1e-5));
}
