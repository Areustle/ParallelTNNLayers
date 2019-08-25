#ifndef CP4CONV2DFORWARD_CUH
#define CP4CONV2DFORWARD_CUH


#include "Tensor.cuh"

float cp4_conv2d_forward_gpu(tensor_shape,
                             const float*,
                             const float*,
                             const float*,
                             const float*,
                             const float*,
                             float*,
                             unsigned PROFCOUNT = 1);


#endif /* CP4CONV2DFORWARD_CUH */
