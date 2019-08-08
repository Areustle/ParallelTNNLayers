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

void cuda_conv2d_cp4_gpu(const float*   In,
                         const unsigned N,
                         const unsigned C,
                         const unsigned H,
                         const unsigned W,
                         const unsigned pad,
                         const float*   FilterK,
                         const float*   FilterC,
                         const float*   FilterH,
                         const float*   FilterW,
                         const unsigned fRank,
                         const unsigned fK,
                         const unsigned fC,
                         const unsigned fH,
                         const unsigned fW,
                         float*         Out);


#endif /* CP4CONV2D_H */
