#ifndef CONV_CUH
#define CONV_CUH

#include "Tensor.cuh"

Tensor conv2d_full_gpu(Tensor const Input, Tensor const Filter, int pad = 1);
Tensor conv2d_full_cpu(Tensor const Input, Tensor const Filter);

void cuda_conv2d_full_gpu(const float* In,
                          const int    N,
                          const int    C,
                          const int    H,
                          const int    W,
                          const int    pad,
                          const float* Filter,
                          const int    fK,
                          const int    fC,
                          const int    fH,
                          const int    fW,
                          float*       Out);

void cuda_conv2d_full_cpu(const float* Input, const float* Filter, int pad);

#endif /* CONV_CUH */
