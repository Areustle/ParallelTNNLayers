#ifndef CP4CONV2D_H
#define CP4CONV2D_H

#include "Tensor.cuh"

Tensor conv2d_cp4_gpu(Tensor const,
                      Tensor const,
                      Tensor const,
                      Tensor const,
                      Tensor const,
                      int pad = 1);

Tensor conv2d_cp4_cpu(Tensor const,
                      Tensor const,
                      Tensor const,
                      Tensor const,
                      Tensor const,
                      int pad = 1);

void cuda_conv2d_cp4_gpu(const float* In,
                         const int    N,
                         const int    C,
                         const int    H,
                         const int    W,
                         const int    pad,
                         const float* FilterK,
                         const float* FilterC,
                         const float* FilterH,
                         const float* FilterW,
                         const int    fRank,
                         const int    fK,
                         const int    fC,
                         const int    fH,
                         const int    fW,
                         float*       Out);


#endif /* CP4CONV2D_H */
