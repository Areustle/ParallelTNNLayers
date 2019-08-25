
#include "Tensor.cuh"

float cp4_conv2d_backward_data_gpu(tensor_shape,
                                   const float*,
                                   const float*,
                                   const float*,
                                   const float*,
                                   const float*,
                                   float*,
                                   unsigned PROFCOUNT = 1);
