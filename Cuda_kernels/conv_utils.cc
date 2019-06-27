#include "conv_utils.h"
#include <cstdlib>

void pad_array(float*  U,
               size_t* dN,
               size_t* dC,
               size_t* dH,
               size_t* dW,
               int     pad) {
  size_t newH   = 2 * pad + *dH;
  size_t newW   = 2 * pad + *dW;
  size_t newLen = *dN * *dC * newH * newW;


  float* newU = (float*)std::malloc(newLen * sizeof(float));
  for (size_t n = 0; n < *dN; ++n)
    for (size_t c = 0; c < *dC; ++c)
      for (size_t h = 0; h < newH; ++h)
        for (size_t w = 0; w < newW; ++w)
          newU[n * (*dC) * newH * newW + c * newH * newW + h * newW + w] = 0.0;

  for (size_t n = 0; n < *dN; ++n)
    for (size_t c = 0; c < *dC; ++c)
      for (size_t h = 0; h < *dH; ++h)
        for (size_t w = 0; w < *dW; ++w)
          newU[(n * (*dC) * newH * newW) + (c * newH * newW) + (h * newW + pad)
               + (w + pad)] = U[(n * (*dC) * (*dH) * (*dW))
                                + (c * (*dH) * (*dW)) + (h * (*dW)) + w];
}


void cpu_imp::conv2d(float*       U, //[N,C,H,W]
                     float*       K, //[C,F,KH,KW]
                     float*       V, //[N,C,H,W]
                     const size_t dN,
                     const size_t dC,
                     const size_t dH,
                     const size_t dW,
                     const size_t dF,
                     const size_t dKH,
                     const size_t dKW) {

  /* zero out the V array */
  for (int n = 0; n < dN; ++n)
    for (int c = 0; c < dC; ++c)
      for (int h = 0; h < dH; ++h)
        for (int w = 0; w < dW; ++w)
          V[n + c * dN + h * dN * dC + w * dN * dC * dH] = 0;

  /* Perform the naive convolution */
  for (int n = 0; n < dN; ++n)
    for (int c = 0; c < dC; ++c)
      for (int h = 0; h < dH; ++h)
        for (int w = 0; w < dW; ++w)
          for (int f = 0; f < dF; ++f)
            for (int kh = 0; kh < dKH; ++kh)
              for (int kw = 0; kw < dKW; ++kw)
                V[(n * dC * dW * dH) + (c * dW * dH) + (h * dW) + w] +=
                    K[(n * dF * dKW * dKH) + (f * dKW * dKW) + (kh * dKW) + kw]
                    * U[(n * dC * dW * dH) + (c * dW * dH) + (h + kh) * dW
                        + (w + kw)];
}
