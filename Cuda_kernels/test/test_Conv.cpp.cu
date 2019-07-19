#define DOCTEST_CONFIG_IMPLEMENTATION_IN_DLL
#include "../../external/doctest/doctest.h"
#include "../Tensor.cuh"
#include "../Utils.cuh"
/* #include "../cp4Conv2d.h" */
#include "../conv.cuh"
#include "../cudnnConv2d.cuh"
#include "../manual.cuh"

TEST_CASE("Convolution test") {

  /* auto K0    = random_fill({ 16, 6 }, -1, 1); */
  /* auto K1    = random_fill({ 16, 6 }, -1, 1); */
  /* auto K2    = random_fill({ 3, 6 }, -1, 1); */
  /* auto K3    = random_fill({ 3, 6 }, -1, 1); */
  /* auto U = random_fill({ 1, 1, 32, 32 }, 0, 1); */
  /* auto K = random_fill({ 1, 1, 3, 3 }, 0, 1); */
  /* /1* auto K     = cp4recom(K0, K1, K2, K3); *1/ */
  /* auto Cudnn    = nn_conv2d(U, K); */
  /* /1* auto cpu_full = conv2d_full_cpu(U, K); *1/ */
  /* /1* for (int i = 0; i < Cudnn.size(); ++i) *1/ */
  /* /1*   REQUIRE(Cudnn[i] == doctest::Approx(cpu_full[i]).epsilon(1e-5)); *1/ */

  /* auto padU = padNCHW(U, 1); */
  /* REQUIRE(padU.size() == 1156); */
  /* REQUIRE(padU.shape[0] == 1); */
  /* REQUIRE(padU.shape[1] == 1); */
  /* REQUIRE(padU.shape[2] == 34); */
  /* REQUIRE(padU.shape[3] == 34); */
  /* for (int i = 0; i<34; ++i) */
  /*   REQUIRE(padU.m_data[i] == 0); */
  /* for (int i = 33*34; i<34*34; ++i) */
  /*   REQUIRE(padU.m_data[i] == 0); */
  /* for (int i = 0; i<34*34; i+=34){ */
  /*   REQUIRE(padU.m_data[i] == 0); */
  /*   REQUIRE(padU.m_data[i+33] == 0); */
  /* } */
  /* auto Full_gpu = conv2d_full_gpu(padU, K); */
  /* REQUIRE(Cudnn.size() == Full_gpu.size()); */
  /* REQUIRE(Full_gpu.shape[0] == 1); */
  /* REQUIRE(Full_gpu.shape[1] == 1); */
  /* REQUIRE(Full_gpu.shape[2] == 32); */
  /* REQUIRE(Full_gpu.shape[3] == 32); */
  /* for (int i = 0; i < Cudnn.size(); ++i) */
  /*   REQUIRE(Cudnn[i] == doctest::Approx(Full_gpu[i]).epsilon(1e-5)); */
}
