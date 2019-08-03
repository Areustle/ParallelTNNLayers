#ifndef NVCONV2D_CUH
#define NVCONV2D_CUH

#include "Tensor.cuh"
#include <cudnn.h>

namespace NV {

  Tensor Conv2dForward(const Tensor, const Tensor);
  float  conv2d_forward_gpu(float* In,
                            int    N,
                            int    C,
                            int    H,
                            int    W,
                            float* Filter,
                            int    fK,
                            int    fH,
                            int    fW,
                            float* Out);

} // namespace NV

#endif /* NVCONV2D_CUH */
