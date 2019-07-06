#ifndef MANUAL_H
#define MANUAL_H

#include "Tensor.h"

Tensor static_cp4_conv2d(Tensor const U, Tensor const K0, Tensor const K1,
                         Tensor const K2, Tensor const K3);

Tensor naive_conv2d(Tensor const U, Tensor const K);

#endif /* MANUAL_H */
