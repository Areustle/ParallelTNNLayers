#ifndef CONV_UTILS_H
#define CONV_UTILS_H

#include "Tensor.h"
#include <cstddef>


Tensor random_fill(size_t N,
                   size_t C,
                   size_t H,
                   size_t W,
                   float  lo = -1.0,
                   float  hi = 1.0);


#endif /* CONV_UTILS_H */
