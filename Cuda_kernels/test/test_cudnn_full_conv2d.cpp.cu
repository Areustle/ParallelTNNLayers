#include "doctest.h"

#include "Tensor.h"
#include "cudnn_full_conv2d.h"

#include <iostream>

TEST_CASE("cudnn_full_conv2d test") {
  Tensor U(1, 1, 32, 32);
  Tensor K(1, 1, 3, 3);
  Tensor U0(1, 1, 32, 32);
  Tensor U1(1, 1, 32, 32);
  Tensor V(1, 1, 32, 32);

  CudnnConv2d conv0(1, 1, 32, 32, 1, 3, 3);
  CudnnConv2d conv1(1, 1, 32, 32, 1, 3, 3);
  CudnnConv2d conv2(1, 1, 32, 32, 1, 3, 3);

  conv0.conv2d(U.m_data, K.m_data, U0.m_data);
  conv0.conv2d(U0.m_data, K.m_data, U1.m_data);
  conv0.conv2d(U1.m_data, K.m_data, V.m_data);

  for (int i = 0; i < V.size(); ++i) {
    CHECK(V[i] == 0);
  }
}
