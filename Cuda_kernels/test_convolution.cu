#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "cudnn_full_conv2d.h"
#include "doctest.h"
#include <cudnn.h>

TEST_CASE( "Check cudnn full convolution" ) {

  const size_t dN = 1, dC = 16, dH = 32, dW = 32, dF = 16, dKH = 3, dKW = 3;
  float *      U, V;
  void*        K;

  cudaMalloc( &U, ( dN * dC * dH * dW ) * sizeof( float ) );
  cudaMalloc( &V, ( dN * dC * dH * dW ) * sizeof( float ) );
  cudaMalloc( &K, ( dC * dF * dKH * dKW ) * sizeof( float ) );

  CHECK(cudnn_imp::conv2d(U, K, V, dN, dC, dH, dW, dF, dKH, dKW)
      == )

  cudaFree( U );
  cudaFree( V );
  cudaFree( K );
}
