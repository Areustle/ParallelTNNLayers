#ifndef CP4CONV2DBACKWARDFILTER_CUH
#define CP4CONV2DBACKWARDFILTER_CUH


#include "Tensor.cuh"

float cp4_conv2d_backward_filter_full_gpu(tensor_shape,
                                          float*,
                                          const float*,
                                          const float*,
                                          unsigned PROFCOUNT = 1);

float cp4_conv2d_backward_filter_t_gpu(tensor_shape,
                                       float*,
                                       const float*,
                                       const float*,
                                       const float*,
                                       const float*,
                                       const float*,
                                       unsigned PROFCOUNT = 1);

float cp4_conv2d_backward_filter_c_gpu(tensor_shape,
                                       float*,
                                       const float*,
                                       const float*,
                                       const float*,
                                       const float*,
                                       const float*,
                                       unsigned PROFCOUNT = 1);

float cp4_conv2d_backward_filter_y_gpu(tensor_shape,
                                       float*,
                                       const float*,
                                       const float*,
                                       const float*,
                                       const float*,
                                       const float*,
                                       unsigned PROFCOUNT = 1);

float cp4_conv2d_backward_filter_x_gpu(tensor_shape,
                                       float*,
                                       const float*,
                                       const float*,
                                       const float*,
                                       const float*,
                                       const float*,
                                       unsigned PROFCOUNT = 1);

#endif /* CP4CONV2DBACKWARDFILTER_CUH */
