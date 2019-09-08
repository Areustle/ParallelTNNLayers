#ifndef NVCONV2D_CUH
#define NVCONV2D_CUH

#include "Tensor.cuh"
#include <cudnn.h>

namespace NV {

  Tensor Conv2dForward(const Tensor, const Tensor, unsigned);
  Tensor Conv2dBackwardData(const Tensor, const Tensor, unsigned);
  Tensor
  Conv2dBackwardFilter(const Tensor, const Tensor, const Tensor, unsigned);

  std::pair<float, unsigned>
  run_convolution(tensor_shape, unsigned PROFCOUNT = 1);

} // namespace NV

#endif /* NVCONV2D_CUH */
