#ifndef CP4CONV2D_H
#define CP4CONV2D_H

#include "Tensor.cuh"

Tensor conv2d_cp4_gpu(Tensor const,
                      Tensor const,
                      Tensor const,
                      Tensor const,
                      Tensor const,
                      int pad = 1);

Tensor conv2d_cp4_cpu(Tensor const,
                      Tensor const,
                      Tensor const,
                      Tensor const,
                      Tensor const,
                      int pad = 1);


#endif /* CP4CONV2D_H */
