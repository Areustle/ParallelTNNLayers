#ifndef CUDNN_FULL_CONV2D_H
#define CUDNN_FULL_CONV2D_H

#include "Tensor.cuh"

#include <cudnn.h>

Tensor nn_conv2d(Tensor const U, Tensor const K);

#endif /* CUDNN_FULL_CONV2D_H */
