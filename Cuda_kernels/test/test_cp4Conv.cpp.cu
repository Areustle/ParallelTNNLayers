#define DOCTEST_CONFIG_IMPLEMENTATION_IN_DLL
#include "../../external/doctest/doctest.h"
#include "../NVConv2d.cuh"
#include "../Tensor.cuh"
#include "../Utils.cuh"
#include "../cp4Conv2d.cuh"
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

using namespace std;

TEST_CASE("Convolution test") {

  static const unsigned n    = 1;
  static const unsigned c    = 32;
  static const unsigned x    = 128;
  static const unsigned pad  = 1;
  static const unsigned k    = 4;
  static const unsigned f    = 3;
  static const unsigned rank = 4;

  Tensor Input = random_fill({ n, c, x, x });
  Tensor K0    = random_fill({ k, rank });
  Tensor K1    = random_fill({ c, rank });
  Tensor K2    = random_fill({ f, rank });
  Tensor K3    = random_fill({ f, rank });
  Tensor K     = cp4recom(K0, K1, K2, K3);

  Tensor Cudnn = NV::Conv2dForward(Input, K, pad);

  Tensor CP4 = CP::Conv2dForward<n, c, x, x, pad, k, f, f, rank>(
      Input, K0, K1, K2, K3);

  REQUIRE(Cudnn.size() == CP4.size());
  REQUIRE(CP4.shape[0] == n);
  REQUIRE(CP4.shape[1] == k);
  REQUIRE(CP4.shape[2] == x);
  REQUIRE(CP4.shape[3] == x);

  /* for (int i = 0; i < Cudnn.size(); ++i) */
  /*   REQUIRE(Cudnn.m_data[i] == doctest::Approx(CP4.m_data[i]).epsilon(1e-5));
   */

  REQUIRE(AllClose(Cudnn, CP4, 1e-5));
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

template<unsigned N,
         unsigned C,
         unsigned H,
         unsigned W,
         unsigned pad,
         unsigned fK,
         unsigned fH,
         unsigned fW,
         unsigned fRank>
void TensorTest() {

  Tensor Input = random_fill({ N, C, H, W });
  Tensor K0    = random_fill({ fK, fRank });
  Tensor K1    = random_fill({ C, fRank });
  Tensor K2    = random_fill({ fH, fRank });
  Tensor K3    = random_fill({ fW, fRank });

  Tensor K     = cp4recom(K0, K1, K2, K3);
  Tensor Cudnn = NV::Conv2dForward(Input, K, pad);

  Tensor CP4 = CP::Conv2dForward<N, C, H, W, pad, fK, fH, fW, fRank>(
      Input, K0, K1, K2, K3);

  REQUIRE(Cudnn.size() == CP4.size());
  REQUIRE(CP4.shape[0] == N);
  REQUIRE(CP4.shape[1] == fK);
  REQUIRE(CP4.shape[2] == H);
  REQUIRE(CP4.shape[3] == W);


  /* for (int i = 0; i < Cudnn.size(); ++i) */
  /*   REQUIRE(Cudnn.m_data[i] == doctest::Approx(CP4.m_data[i]).epsilon(1e-3)); */

  string error_message = "Incorrect result with "
    + to_string(N) + " , "
    + to_string(C) + " , "
    + to_string(H) + " , "
    + to_string(W) + " , "
    + to_string(pad) + " , "
    + to_string(fK) + " , "
    + to_string(fH) + " , "
    + to_string(fW) + " , "
    + to_string(fRank);

  CHECK_MESSAGE(AllClose(Cudnn, CP4, 1e-5), error_message);
}

template<unsigned rank> void test_helper() {

  test_helper<rank - 1>();

  // Batch Size
  TensorTest<1, 3, 512, 512, 1, 1, 3, 3, rank>();
  TensorTest<2, 3, 512, 512, 1, 1, 3, 3, rank>();
  TensorTest<4, 3, 512, 512, 1, 1, 3, 3, rank>();
  TensorTest<8, 3, 512, 512, 1, 1, 3, 3, rank>();
  TensorTest<16, 3, 512, 512, 1, 1, 3, 3, rank>();
  TensorTest<32, 3, 512, 512, 1, 1, 3, 3, rank>();
  TensorTest<64, 3, 512, 512, 1, 1, 3, 3, rank>();
  TensorTest<128, 3, 512, 512, 1, 1, 3, 3, rank>();

  // Image Size
  TensorTest<1, 3, 32, 32, 1, 1, 3, 3, rank>();     // 1
  TensorTest<1, 3, 64, 64, 1, 1, 3, 3, rank>();     // 2
  TensorTest<1, 3, 128, 128, 1, 1, 3, 3, rank>();   // 3
  TensorTest<1, 3, 256, 256, 1, 1, 3, 3, rank>();   // 4
  TensorTest<1, 3, 512, 512, 1, 1, 3, 3, rank>();   // 5
  TensorTest<1, 3, 1024, 1024, 1, 1, 3, 3, rank>(); // 6
  TensorTest<1, 3, 2048, 2048, 1, 1, 3, 3, rank>(); // 7
  TensorTest<1, 3, 4096, 4096, 1, 1, 3, 3, rank>(); // 8

  // Channel Depth
  TensorTest<1, 1, 512, 512, 1, 1, 3, 3, rank>();
  TensorTest<1, 2, 512, 512, 1, 1, 3, 3, rank>();
  TensorTest<1, 4, 512, 512, 1, 1, 3, 3, rank>();
  TensorTest<1, 8, 512, 512, 1, 1, 3, 3, rank>();
  TensorTest<1, 16, 512, 512, 1, 1, 3, 3, rank>();
  TensorTest<1, 32, 512, 512, 1, 1, 3, 3, rank>();
  TensorTest<1, 64, 512, 512, 1, 1, 3, 3, rank>();
  TensorTest<1, 128, 512, 512, 1, 1, 3, 3, rank>();

  // Output Channels
  TensorTest<1, 3, 512, 512, 1, 1,   3, 3, rank>();
  TensorTest<1, 3, 512, 512, 1, 2,   3, 3, rank>();
  TensorTest<1, 3, 512, 512, 1, 4,   3, 3, rank>();
  TensorTest<1, 3, 512, 512, 1, 8,   3, 3, rank>();
  TensorTest<1, 3, 512, 512, 1, 16,  3, 3, rank>();
  TensorTest<1, 3, 512, 512, 1, 32,  3, 3, rank>();
  TensorTest<1, 3, 512, 512, 1, 64,  3, 3, rank>();
  TensorTest<1, 3, 512, 512, 1, 128, 3, 3, rank>();

  // Filter Size
  TensorTest<1, 3, 512, 512, 1, 1, 3, 3,   rank>();
  TensorTest<1, 3, 512, 512, 2, 1, 5, 5,   rank>();
  TensorTest<1, 3, 512, 512, 3, 1, 7, 7,   rank>();
  TensorTest<1, 3, 512, 512, 4, 1, 9, 9,   rank>();
  TensorTest<1, 3, 512, 512, 5, 1, 11, 11, rank>();
  TensorTest<1, 3, 512, 512, 6, 1, 13, 13, rank>();
  TensorTest<1, 3, 512, 512, 7, 1, 15, 15, rank>();
  TensorTest<1, 3, 512, 512, 8, 1, 17, 17, rank>();

  TensorTest<32, 32, 512, 512, 1, 32, 3, 3, rank>();
}
template<> void test_helper<0>() {}


TEST_CASE("Extended Convolution Test") { test_helper<16>(); }
