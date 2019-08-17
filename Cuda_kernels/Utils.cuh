#ifndef CONV_UTILS_H
#define CONV_UTILS_H

#include "Tensor.cuh"

#include <cstddef>
#include <initializer_list>

Tensor random_fill(std::initializer_list<unsigned> lst);

Tensor cp4recom(Tensor A, Tensor B, Tensor C, Tensor D);

bool AllClose(Tensor, Tensor, float tolerance=1e-3);

#endif /* CONV_UTILS_H */
