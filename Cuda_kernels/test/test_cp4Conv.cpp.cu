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
  unsigned c    = 32;
  unsigned x    = 128;
  unsigned pad  = 1;
  unsigned k    = 4;
  unsigned f    = 3;
  unsigned rank = 4;

  SUBCASE("Forward Convolution Data test") {

    auto Input = random_fill({ n, c, x, x });
    auto FT    = random_fill({ k, rank });
    auto FC    = random_fill({ c, rank });
    auto FY    = random_fill({ f, rank });
    auto FX    = random_fill({ f, rank });
    auto K     = cp4recom(FT, FC, FY, FX);

    auto Cudnn = NV::Conv2dForward(Input, K, pad);
    auto CP4   = CP::Conv2dForward(Input, FT, FC, FY, FX, pad);

    REQUIRE(Cudnn.size() == CP4.size());
    REQUIRE(CP4.shape[0] == n);
    REQUIRE(CP4.shape[1] == k);
    REQUIRE(CP4.shape[2] == x);
    REQUIRE(CP4.shape[3] == x);
    REQUIRE(AllClose(Cudnn, CP4, 1e-5));
  }

  SUBCASE("Backward Convolution Data test") {

    auto dV = random_fill({ n, k, x, x });
    auto FT = random_fill({ k, rank });
    auto FC = random_fill({ c, rank });
    auto FY = random_fill({ f, rank });
    auto FX = random_fill({ f, rank });
    auto K  = cp4recom(FT, FC, FY, FX);

    auto Cudnn = NV::Conv2dBackwardData(dV, K, pad);
    auto CP4   = CP::Conv2dBackwardData(dV, FT, FC, FY, FX, pad);

    REQUIRE(Cudnn.size() == n * c * x * x);
    REQUIRE(Cudnn.size() == CP4.size());
    REQUIRE(CP4.shape[0] == n);
    REQUIRE(CP4.shape[1] == c);
    REQUIRE(CP4.shape[2] == x);
    REQUIRE(CP4.shape[3] == x);

    REQUIRE(AllClose(Cudnn, CP4, 1e-5));
  }
}


TEST_CASE("Extended Convolution Test") {

  std::vector<std::string> tensor_list{
    "Cuda_kernels/bench/tensors.txt",
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

      SUBCASE("Forward Convolution Test") {

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

      SUBCASE("Forward Convolution Test") {

        auto Cudnn = NV::Conv2dBackwardData(Input, FF, pad);
        auto CP4   = CP::Conv2dBackwardData(Input, FC, FT, FY, FX, pad);

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
}
