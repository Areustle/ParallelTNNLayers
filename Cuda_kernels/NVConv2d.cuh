#ifndef NVCONV2D_CUH
#define NVCONV2D_CUH

#include "Tensor.cuh"
#include <cudnn.h>

namespace NV {

  Tensor Conv2dForward(const Tensor, const Tensor, unsigned);
  void   conv2d_forward_gpu(float*   In,
                            unsigned N,
                            unsigned C,
                            unsigned H,
                            unsigned W,
                            unsigned pad,
                            float*   Filter,
                            unsigned fK,
                            unsigned fH,
                            unsigned fW,
                            float*   Out);

} // namespace NV

#endif /* NVCONV2D_CUH */
