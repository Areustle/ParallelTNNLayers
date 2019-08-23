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

/* TEST_CASE("Convolution test") {*/

/*   unsigned n    = 1;*/
/*   unsigned c    = 32;*/
/*   unsigned x    = 128;*/
/*   unsigned pad  = 1;*/
/*   unsigned k    = 4;*/
/*   unsigned f    = 3;*/
/*   unsigned rank = 4;*/

/*   auto Input = random_fill({ n, c, x, x });*/
/*   auto K0    = random_fill({ k, rank });*/
/*   auto K1    = random_fill({ c, rank });*/
/*   auto K2    = random_fill({ f, rank });*/
/*   auto K3    = random_fill({ f, rank });*/
/*   auto K     = cp4recom(K0, K1, K2, K3);*/

/*   auto Cudnn = NV::Conv2dForward(Input, K, pad);*/

/*   auto CP4 = CP::Conv2dForward(Input, K0, K1, K2, K3, pad);*/

/*   REQUIRE(Cudnn.size() == CP4.size());*/
/*   REQUIRE(CP4.shape[0] == n);*/
/*   REQUIRE(CP4.shape[1] == k);*/
/*   REQUIRE(CP4.shape[2] == x);*/
/*   REQUIRE(CP4.shape[3] == x);*/

/*   REQUIRE(AllClose(Cudnn, CP4, 1e-5));*/
/* }*/

/* TEST_CASE("Extended Convolution Test") { */

/*   std::vector<std::string> tensor_list{ */
/*     "Cuda_kernels/bench/tensors.txt", */
/*     "Cuda_kernels/bench/tensors_batch_size.txt", */
/*     "Cuda_kernels/bench/tensors_channel_depth.txt", */
/*     "Cuda_kernels/bench/tensors_image_size.txt", */
/*     "Cuda_kernels/bench/tensors_filter_count.txt", */
/*     "Cuda_kernels/bench/tensors_filter_size.txt", */
/*     "Cuda_kernels/bench/tensors_all_scales.txt", */
/*   }; */

/*   for (auto t : tensor_list) { */
/*     ifstream tensors(t); */

/*     REQUIRE(tensors.is_open()); */
/*     string line; */
/*     while (getline(tensors, line)) { */

/*       if (line[0] == '#' || line.empty()) continue; */

/*       stringstream line_sm(line); */
/*       unsigned     N, H, W, C, pad, fK, fH, fW, fRank; */
/*       line_sm >> N >> C >> H >> W >> pad >> fK >> fH >> fW >> fRank; */

/*       auto Input   = random_fill({ N, C, H, W }); */
/*       auto FilterK = random_fill({ fK, fRank }); */
/*       auto FilterC = random_fill({ C, fRank }); */
/*       auto FilterH = random_fill({ fH, fRank }); */
/*       auto FilterW = random_fill({ fW, fRank }); */
/*       auto Filter  = cp4recom(FilterK, FilterC, FilterH, FilterW); */

/*       auto Cudnn = NV::Conv2dForward(Input, Filter, pad); */

/*       auto CP4 */
/*           = CP::Conv2dForward(Input, FilterK, FilterC, FilterH, FilterW, pad); */

/*       REQUIRE(Cudnn.size() == CP4.size()); */
/*       REQUIRE(CP4.shape[0] == N); */
/*       REQUIRE(CP4.shape[1] == fK); */
/*       REQUIRE(CP4.shape[2] == H); */
/*       REQUIRE(CP4.shape[3] == W); */
/*       REQUIRE_MESSAGE( */
/*           /1* Cudnn.m_data[i] *1/ */
/*           /1*     == doctest::Approx(CP4.m_data[i]).epsilon(1e-5), *1/ */
/*           AllClose(Cudnn, CP4, 1e-5), */
/*           "Incorrect result with " << line << " Parsed as " << N << "," << C */
/*                                    << "," << H << "," << W << "," << pad << "," */
/*                                    << fK << "," << fH << "," << fW << "," */
/*                                    << fRank); */
/*     } */
/*   } */
/* } */
