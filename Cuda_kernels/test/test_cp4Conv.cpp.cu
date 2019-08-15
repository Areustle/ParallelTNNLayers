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

  unsigned n    = 1;
  unsigned c    = 16;
  unsigned x    = 32;
  unsigned pad  = 1;
  unsigned k    = 16;
  unsigned rank = 16;
  unsigned f    = 3;

  auto Input = random_fill({ n, c, x, x }, 0, 1);
  auto K0    = random_fill({ k, rank }, 0, 1);
  auto K1    = random_fill({ c, rank }, 0, 1);
  auto K2    = random_fill({ f, rank }, 0, 1);
  auto K3    = random_fill({ f, rank }, 0, 1);
  auto K     = cp4recom(K0, K1, K2, K3);

  auto Cudnn = NV::Conv2dForward(Input, K, pad);

  auto CP4Conv2d = conv2d_cp4_gpu(Input, K0, K1, K2, K3, pad);

  REQUIRE(Cudnn.size() == CP4Conv2d.size());
  REQUIRE(CP4Conv2d.shape[0] == n);
  REQUIRE(CP4Conv2d.shape[1] == k);
  REQUIRE(CP4Conv2d.shape[2] == x);
  REQUIRE(CP4Conv2d.shape[3] == x);
  for (int i = 0; i < Cudnn.size(); ++i)
    REQUIRE(Cudnn.m_data[i]
            == doctest::Approx(CP4Conv2d.m_data[i]).epsilon(1e-5));
}

TEST_CASE("Extended Convolution Test") {

  std::vector<std::string> tensor_list{
    "Cuda_kernels/bench/tensors.txt"
    /* "Cuda_kernels/bench/tensors_batch_size.txt", */
    /* "Cuda_kernels/bench/tensors_channel_depth.txt", */
    /* "Cuda_kernels/bench/tensors_image_size.txt", */
    /* "Cuda_kernels/bench/tensors_filter_count.txt", */
    /* "Cuda_kernels/bench/tensors_filter_size.txt", */
  };

  Tensor Input{ 0, 0, 0, 0 };
  Tensor Cudnn{ 0, 0, 0, 0 };
  Tensor Filter{ 0, 0, 0, 0 };

  for (auto t : tensor_list) {
    ifstream tensors(t);

    REQUIRE(tensors.is_open());
    string line;
    while (getline(tensors, line)) {

      if (line[0] == '#' || line.empty()) continue;

      stringstream line_sm(line);
      unsigned     N, H, W, C, pad, fK, fH, fW, fRank;
      line_sm >> N >> C >> H >> W >> pad >> fK >> fH >> fW >> fRank;

      auto FilterK = random_fill({ fK, fRank }, 0, 1);
      auto FilterC = random_fill({ C, fRank }, 0, 1);
      auto FilterH = random_fill({ fH, fRank }, 0, 1);
      auto FilterW = random_fill({ fW, fRank }, 0, 1);

      /* if (Cudnn.shape[0] != N &&   // */
      /*     Cudnn.shape[1] != C &&   // */
      /*     Cudnn.shape[2] != H &&   // */
      /*     Cudnn.shape[3] != W &&   // */
      /*     Filter.shape[0] != fK && // */
      /*     Filter.shape[2] != fH && // */
      /*     Filter.shape[3] != fW) { */
        Input   = random_fill({ N, C, H, W }, 0, 1);
        Filter = cp4recom(FilterK, FilterC, FilterH, FilterW);
        Cudnn  = NV::Conv2dForward(Input, Filter, pad);
      /* } */

      auto CP4Conv2dGPU
          = conv2d_cp4_gpu(Input, FilterK, FilterC, FilterH, FilterW, pad);

      REQUIRE(Cudnn.size() == CP4Conv2dGPU.size());
      REQUIRE(CP4Conv2dGPU.shape[0] == N);
      REQUIRE(CP4Conv2dGPU.shape[1] == fK);
      REQUIRE(CP4Conv2dGPU.shape[2] == H);
      REQUIRE(CP4Conv2dGPU.shape[3] == W);
      for (int i = 0; i < Cudnn.size(); ++i)
        REQUIRE_MESSAGE(
            Cudnn.m_data[i]
                == doctest::Approx(CP4Conv2dGPU.m_data[i]).epsilon(1e-5),

            "Incorrect result with "
                << line << " Parsed as " << N << "," << C << "," << H << ","
                << W << "," << pad << "," << fK << "," << fH << "," << fW << ","
                << fRank);
    }
  }
}
