#ifndef CP4CONV2D_H
#define CP4CONV2D_H

#include "Tensor.h"

Tensor cp4conv2d(Tensor const U,
                 Tensor const K0,
                 Tensor const K1,
                 Tensor const K2,
                 Tensor const K3);

#endif /* CP4CONV2D_H */
