#define DOCTEST_CONFIG_IMPLEMENTATION_IN_DLL
#include "../../external/doctest/doctest.h"
#include "../Tensor.h"
#include "../Utils.h"
#include "../cp4Conv2d.h"
#include "../cudnnConv2d.h"
#include "../manual.h"

TEST_CASE("Convolution test") {

  auto K0    = random_fill({ 16, 6 }, -1, 1);
  auto K1    = random_fill({ 16, 6 }, -1, 1);
  auto K2    = random_fill({ 3, 6 }, -1, 1);
  auto K3    = random_fill({ 3, 6 }, -1, 1);
  auto U     = random_fill({ 1, 16, 32, 32 }, 0, 1);
  auto K     = cp4recom(K0, K1, K2, K3);
  auto Cudnn = nn_conv2d(U, K);
  auto Hand  = conv2d_full_cpu(U, K);
  auto Full_gpu  = conv2d_full_gpu(U, K);
  /* auto Tvm   = static_cp4_conv2d(U, K1, K2, K3, K0); */
  /* auto HandCp4  = conv2d_cp4_cpu(U, K0, K1, K2, K3); */

  // Handwritten Full
  REQUIRE(Cudnn.size() == Hand.size());
  for (int i = 0; i < Cudnn.size(); ++i)
    REQUIRE(Cudnn[i] == doctest::Approx(Hand[i]).epsilon(1e-5));
  REQUIRE(Cudnn.size() == Full_gpu.size());
  for (int i = 0; i < Cudnn.size(); ++i)
    REQUIRE(Cudnn[i] == doctest::Approx(Full_gpu[i]).epsilon(1e-5));

  /* // TVM CP4 */
  /* REQUIRE(Cudnn.size() == Tvm.size()); */
  /* for (int i = 0; i < Cudnn.size(); ++i) */
  /*   REQUIRE(Cudnn[i] == doctest::Approx(Tvm[i]).epsilon(1e-4)); */

  /* // Handwritten CP4 */
  /* REQUIRE(Cudnn.size() == HandCp4.size()); */
  /* for (int i = 0; i < Cudnn.size(); ++i) */
  /*   REQUIRE(Cudnn[i] == doctest::Approx(HandCp4[i]).epsilon(1e-4)); */
}
