#include "conv_utils.h"
#include <cstdlib>

void pad_array( float*  U,
                size_t* dN,
                size_t* dC,
                size_t* dH,
                size_t* dW,
                int     pad ) {
  size_t newH   = pad + *dH;
  size_t newW   = pad + *dW;
  size_t newLen = *dN * *dC * newH * newW;

  float* newU = (float*)malloc( newLen * sizeof( float ) );
}
