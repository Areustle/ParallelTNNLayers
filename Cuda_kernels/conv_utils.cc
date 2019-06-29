#include "conv_utils.h"
#include <cstdlib>

void pad_array(float*  U,
               size_t* N,
               size_t* C,
               size_t* H,
               size_t* W,
               int     pad) {
  size_t newH   = 2 * pad + *H;
  size_t newW   = 2 * pad + *W;
  size_t newLen = *N * *C * newH * newW;


  float* newU = (float*)std::malloc(newLen * sizeof(float));
  for (size_t n = 0; n < *N; ++n)
    for (size_t c = 0; c < *C; ++c)
      for (size_t h = 0; h < newH; ++h)
        for (size_t w = 0; w < newW; ++w)
          newU[n * (*C) * newH * newW + c * newH * newW + h * newW + w] = 0.0;

  for (size_t n = 0; n < *N; ++n)
    for (size_t c = 0; c < *C; ++c)
      for (size_t h = 0; h < *H; ++h)
        for (size_t w = 0; w < *W; ++w)
          newU[(n * (*C) * newH * newW) + (c * newH * newW) + (h * newW + pad)
               + (w + pad)] = U[(n * (*C) * (*H) * (*W))
                                + (c * (*H) * (*W)) + (h * (*W)) + w];
}


void cpu_imp::conv2d(float*       U, //[N,C,H,W]
                     float*       K, //[C,F,Y,X]
                     float*       V, //[N,C,H,W]
                     const size_t N,
                     const size_t C,
                     const size_t H,
                     const size_t W,
                     const size_t F,
                     const size_t Y,
                     const size_t X) {

  /* zero out the V array */
  for (int n = 0; n < N; ++n)
    for (int c = 0; c < C; ++c)
      for (int h = 0; h < H; ++h)
        for (int w = 0; w < W; ++w)
          V[n + c * N + h * N * C + w * N * C * H] = 0;

  /* Perform the naive convolution */
  for (int n = 0; n < N; ++n)
  for (int c = 0; c < C; ++c)
  for (int h = 0; h < H; ++h)
  for (int w = 0; w < W; ++w)
  for (int f = 0; f < F; ++f)
  for (int y = 0; y < Y; ++y)
  for (int x = 0; x < X; ++x){
    V[n*C*H*W + c*H*W + h*W + w] +=
    K[n*F*Y*X + f*Y*X + y*X + x] *
    U[n*C*W*H + c*W*H + (h+y)*W + (w+x)];
  }
}
