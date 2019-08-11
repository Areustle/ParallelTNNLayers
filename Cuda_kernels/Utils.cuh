#ifndef CONV_UTILS_H
#define CONV_UTILS_H

#include "Tensor.cuh"

#include <cstddef>
#include <initializer_list>

Tensor random_fill(std::initializer_list<unsigned> lst,
                   float                           lo = -1.0,
                   float                           hi = 1.0);

Tensor cp4recom(Tensor A, Tensor B, Tensor C, Tensor D);

Tensor padNCHW(Tensor, unsigned);

#endif /* CONV_UTILS_H */
