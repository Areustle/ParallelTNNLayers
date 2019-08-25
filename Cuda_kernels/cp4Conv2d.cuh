#ifndef CP4CONV2D_H
#define CP4CONV2D_H

#include "Tensor.cuh"

namespace CP {

  Tensor Conv2dForward(Tensor const,
                       Tensor const,
                       Tensor const,
                       Tensor const,
                       Tensor const,
                       unsigned pad = 1);

  Tensor Conv2dBackwardData(Tensor const,
                            Tensor const,
                            Tensor const,
                            Tensor const,
                            Tensor const,
                            unsigned pad = 1);

  Tensor Conv2dBackwardFilter(Tensor const,
                              Tensor const,
                              Tensor const,
                              Tensor const,
                              Tensor const,
                              unsigned pad = 1);

  float run_convolution(tensor_shape, unsigned PROFCOUNT = 1);

} // namespace CP


#endif /* CP4CONV2D_H */
