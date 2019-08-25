#define DOCTEST_CONFIG_IMPLEMENTATION_IN_DLL
#include "../../external/doctest/doctest.h"
#include "../NVConv2d.cuh"
#include "../Utils.cuh"
/* #include <iostream> */

/* using namespace std; */

TEST_CASE("cudnn_full_conv2d test") {
  Tensor K = { 1, 4, 3, 3 };
  for (int i = 0; i < K.size(); ++i) REQUIRE(K.m_data[i] == 0);

  Tensor U = random_fill({ 1, 4, 32, 32 });

  for (int i = 0; i < U.size(); ++i) REQUIRE(U.m_data[i] != 0);
  for (int i = 0; i < K.size(); ++i) REQUIRE(K.m_data[i] == 0);

  auto V = NV::Conv2dForward(U, K, 1);

  for (int i = 0; i < V.size(); ++i) REQUIRE(V.m_data[i] == 0);
}

TEST_CASE("cudnn backward data test") {

  Tensor U = random_fill({ 16, 1, 32, 32 });
  Tensor K = { 1, 4, 3, 3 };

  for (int i = 0; i < U.size(); ++i) REQUIRE(U.m_data[i] != 0);
  for (int i = 0; i < K.size(); ++i) REQUIRE(K.m_data[i] == 0);

  auto V = NV::Conv2dBackwardData(U, K, 1);

  CHECK(V.size() == (16 * 4 * 32 * 32));
  for (int i = 0; i < V.size(); ++i) REQUIRE(V.m_data[i] == 0);
}

TEST_CASE("cudnn backward filter test") {

  Tensor U  = random_fill({ 16, 4, 32, 32 });
  Tensor dU = random_fill({ 16, 1, 32, 32 });
  Tensor K  = { 1, 4, 3, 3 };

  for (int i = 0; i < U.size(); ++i) REQUIRE(U.m_data[i] != 0);
  for (int i = 0; i < K.size(); ++i) REQUIRE(K.m_data[i] == 0);

  auto V = NV::Conv2dBackwardFilter(U, dU, K, 1);

  CHECK(V.size() == (1 * 4 * 3 * 3));
  for (int i = 0; i < V.size(); ++i) REQUIRE(V.m_data[i] != 0);
}

/* Tensor conv2d_data_grad(Tensor const dU, Tensor const K, const unsigned pad) { */

/*   const unsigned N  = dU.shape[0]; */
/*   const unsigned T  = dU.shape[1]; */
/*   const unsigned H  = dU.shape[2]; */
/*   const unsigned W  = dU.shape[3]; */
/*   /1* const unsigned FT = K.shape[0]; *1/ */
/*   const unsigned C = K.shape[1]; */
/*   const unsigned Y = K.shape[2]; */
/*   const unsigned X = K.shape[3]; */
/*   const unsigned sH = H+2*pad; */
/*   const unsigned sW = W+2*pad; */

/*   Tensor R = {N, C, H, W}; */
/*   Tensor shared_mem = {N, T, sH, sW}; */

/*   for (int n = 0; n < N; ++n) */
/*   for (int t = 0; t < T; ++t) */
/*   for (int h = 0; h < sH; ++h) */
/*   for (int w = 0; w < sW; ++w) */
/*     shared_mem.m_data[n*T*sH*sW + t*sH*sW + h*sW + w] */
/*       = (h >= pad */
/*           && h < H+pad */
/*           && w >= pad */
/*           && w < W+pad) */
/*       ? dU.m_data[n*T*H*W + t*H*W + (h-pad)*W + (w-pad)] */
/*       : 0.0f; */

/*   for (int n = 0; n < N; ++n) */
/*   for (int c = 0; c < C; ++c) */
/*   for (int h = 0; h < H; ++h) */
/*   for (int w = 0; w < W; ++w) */
/*   for (int t = 0; t < T; ++t) */
/*   for (int y = 0; y < Y; ++y) */
/*   for (int x = 0; x < X; ++x) */
/*     R.m_data[n*C*H*W + c*H*W + h*W + w] */
/*       += shared_mem.m_data[n*T*sH*sW + t*sH*sW + (h+y)*sW + (w+x)] */
/*       * K.m_data[t*C*Y*X + c*Y*X + (Y-1-y)*X + (X-1-x)]; */

/*   return R; */
/* } */

/* TEST_CASE("cpu backward filter") { */

/*   Tensor dU = random_fill({ 2, 4, 32, 32 }); */
/*   Tensor K  = random_fill({ 4, 2, 3, 3 }); */
/*   unsigned pad = 1; */

/*   auto V    = NV::Conv2dBackwardData(dU, K, pad); */
/*   auto MINE = conv2d_data_grad(dU, K, pad); */

/*   CHECK(V.size() == dU.shape[0] * K.shape[1] * dU.shape[2] * dU.shape[3]); */
/*   CHECK(MINE.size() == V.size()); */

/*   for (int i = 0; i < V.size(); ++i) */
/*     CHECK(V.m_data[i] == doctest::Approx(MINE.m_data[i]).epsilon(1e-5)); */

/*   REQUIRE(AllClose(V, MINE, 1e-5)); */
/* } */
