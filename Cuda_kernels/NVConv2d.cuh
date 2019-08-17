#ifndef NVCONV2D_CUH
#define NVCONV2D_CUH

#include "Tensor.cuh"
#include <cudnn.h>

namespace NV {

  Tensor Conv2dForward(const Tensor, const Tensor, unsigned);

  float run_convolution(tensor_shape, unsigned PROFCOUNT = 1);

} // namespace NV

#endif /* NVCONV2D_CUH */
