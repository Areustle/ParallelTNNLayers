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

TEST_CASE("Single Convolution Kernel test") {

  unsigned n    = 1;
  unsigned c    = 48;
  unsigned x    = 55;
  unsigned pad  = 2;
  unsigned t    = 256;
  unsigned f    = 5;
  unsigned rank = 8;

  SUBCASE("Forward Convolution Data test") {

    auto Input = random_fill({ n, c, x, x });
    auto FT    = random_fill({ t, rank });
    auto FC    = random_fill({ c, rank });
    auto FY    = random_fill({ f, rank });
    auto FX    = random_fill({ f, rank });
    auto K     = cp4recom(FT, FC, FY, FX);

    auto Cudnn = NV::Conv2dForward(Input, K, pad);
    auto CP4   = CP::Conv2dForward(Input, FT, FC, FY, FX, pad);

    REQUIRE(Cudnn.size() == CP4.size());
    REQUIRE(CP4.shape[0] == n);
    REQUIRE(CP4.shape[1] == t);
    REQUIRE(CP4.shape[2] == x);
    REQUIRE(CP4.shape[3] == x);

    for (int i = 0; i < CP4.size(); ++i)
      REQUIRE(Cudnn.m_data[i] == doctest::Approx(CP4.m_data[i]).epsilon(1e-5));

    REQUIRE(AllClose(Cudnn, CP4, 1e-5));
  }
}


TEST_CASE("Extended Convolution Test") {

  std::vector<std::string> tensor_list{
    "Cuda_kernels/bench/tensors_alexnet.txt",
    "Cuda_kernels/bench/tensors_batch_size.txt",
    "Cuda_kernels/bench/tensors_channel_depth.txt",
    "Cuda_kernels/bench/tensors_image_size.txt",
    "Cuda_kernels/bench/tensors_filter_count.txt",
    "Cuda_kernels/bench/tensors_filter_size.txt",
    "Cuda_kernels/bench/tensors_all_scales.txt",
  };

  for (auto t : tensor_list) {
    ifstream tensors(t);

    REQUIRE(tensors.is_open());
    string line;
    while (getline(tensors, line)) {

      if (line[0] == '#' || line.empty()) continue;

      stringstream line_sm(line);
      unsigned     N, H, W, C, pad, T, Y, X, fRank;
      line_sm >> N >> C >> H >> W >> pad >> T >> Y >> X >> fRank;

      auto Input = random_fill({ N, C, H, W });
      auto FT    = random_fill({ T, fRank });
      auto FC    = random_fill({ C, fRank });
      auto FY    = random_fill({ Y, fRank });
      auto FX    = random_fill({ X, fRank });
      auto FF    = cp4recom(FT, FC, FY, FX);

      auto Cudnn = NV::Conv2dForward(Input, FF, pad);
      auto CP4   = CP::Conv2dForward(Input, FT, FC, FY, FX, pad);

      REQUIRE(Cudnn.size() == CP4.size());
      REQUIRE(CP4.shape[0] == N);
      REQUIRE(CP4.shape[1] == T);
      REQUIRE(CP4.shape[2] == H);
      REQUIRE(CP4.shape[3] == W);
      REQUIRE_MESSAGE(AllClose(Cudnn, CP4, 1e-5),
                      "Incorrect result with "
                          << line << " Parsed as " << N << ","  //
                          << C << "," << H << "," << W << ","   //
                          << pad << "," << T << "," << Y << "," //
                          << X << "," << fRank);


    }
  }
}
