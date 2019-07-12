#ifndef CP4CONV2D_H
#define CP4CONV2D_H

#include "Tensor.h"

Tensor conv2d_cp4_cpu(Tensor const U,
                      Tensor const K0,
                      Tensor const K1,
                      Tensor const K2,
                      Tensor const K3);

Tensor conv2d_cp4_gpu(Tensor const U,
                      Tensor const K0,
                      Tensor const K1,
                      Tensor const K2,
                      Tensor const K3);

Tensor conv2d_full_cpu(Tensor const Input, Tensor const Filter);
Tensor conv2d_full_gpu(Tensor const Input, Tensor const Filter);

#endif /* CP4CONV2D_H */
