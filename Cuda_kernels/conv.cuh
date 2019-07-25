#ifndef CONV_CUH
#define CONV_CUH

#include "Tensor.cuh"

Tensor conv2d_full_gpu(Tensor const Input, Tensor const Filter, int pad=1);
Tensor conv2d_full_cpu(Tensor const Input, Tensor const Filter);

#endif /* CONV_CUH */
