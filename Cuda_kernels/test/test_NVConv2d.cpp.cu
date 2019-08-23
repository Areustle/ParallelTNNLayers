#define DOCTEST_CONFIG_IMPLEMENTATION_IN_DLL
#include "../../external/doctest/doctest.h"
#include "../NVConv2d.cuh"
#include "../Utils.cuh"

/* TEST_CASE("cudnn_full_conv2d test") { */
/*   Tensor K = { 1, 4, 3, 3 }; */
/*   for (int i = 0; i < K.size(); ++i) REQUIRE(K.m_data[i] == 0); */

/*   Tensor U = random_fill({ 1, 4, 32, 32 }); */

/*   for (int i = 0; i < U.size(); ++i) REQUIRE(U.m_data[i] != 0); */
/*   for (int i = 0; i < K.size(); ++i) REQUIRE(K.m_data[i] == 0); */

/*   auto V = NV::Conv2dForward(U, K, 1); */

/*   for (int i = 0; i < V.size(); ++i) REQUIRE(V.m_data[i] == 0); */
/* } */

/* TEST_CASE("cudnn backward data test") { */

/*   Tensor U = random_fill({ 16, 1, 32, 32 }); */
/*   Tensor K = { 1, 4, 3, 3 }; */

/*   for (int i = 0; i < U.size(); ++i) REQUIRE(U.m_data[i] != 0); */
/*   for (int i = 0; i < K.size(); ++i) REQUIRE(K.m_data[i] == 0); */

/*   auto V = NV::Conv2dBackwardData(U, K, 1); */

/*   CHECK(V.size() == (16 * 4 * 32 * 32)); */
/*   for (int i = 0; i < V.size(); ++i) REQUIRE(V.m_data[i] == 0); */
/* } */

/* TEST_CASE("cudnn backward filter test") { */

/*   Tensor U  = random_fill({ 16, 4, 32, 32 }); */
/*   Tensor dU = random_fill({ 16, 1, 32, 32 }); */
/*   Tensor K  = { 1, 4, 3, 3 }; */

/*   for (int i = 0; i < U.size(); ++i) REQUIRE(U.m_data[i] != 0); */
/*   for (int i = 0; i < K.size(); ++i) REQUIRE(K.m_data[i] == 0); */

/*   auto V = NV::Conv2dBackwardFilter(U, dU, K, 1); */

/*   CHECK(V.size() == (1 * 4 * 3 * 3)); */
/*   for (int i = 0; i < V.size(); ++i) REQUIRE(V.m_data[i] != 0); */
/* } */

Tensor conv2d_data_grad(Tensor const dU, Tensor const K) {

  const unsigned H  = dU.shape[2];
  const unsigned W  = dU.shape[3];
  const unsigned FH = K.shape[2];
  const unsigned FW = K.shape[3];

  Tensor R = {H, W};
  /* Tensor padU = {H+1, W+1}; */

  /* for (int  i = 0;  i <  H;  ++i) */
  /* for (int  j = 0;  j <  W;  ++j) */
  /*   padU.m_data[i*(W+2)+j] = (j >= 1 && j < H + 1 && i >= 1 && i < W+1) */ 
  /*     ? dU[j] */

  for (int  i = 0;  i <  H;  ++i)
  for (int  j = 0;  j <  W;  ++j){
    for (int fh = 0; fh < FH; --fh)
    for (int fw = 0; fw < FW; --fw)
      R.m_data[i*W + j]
        += dU.m_data[(i+fh)*W + (j+fw)]
        * K.m_data[(FH-1-fh)*FW + (FW-1-fw)];
  }

  return R;
}

TEST_CASE("cpu backward filter") {

  Tensor dU = random_fill({ 1, 1, 32, 32 });
  Tensor K  = random_fill({ 1, 1, 3, 3 });

  auto V    = NV::Conv2dBackwardData(dU, K, 1);
  auto MINE = conv2d_data_grad(dU, K);

  REQUIRE(AllClose(V, MINE, 1e-5));
}
