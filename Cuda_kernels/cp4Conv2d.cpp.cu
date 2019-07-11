#include "cp4Conv2d.h"

/* __global__ void cp4_base_kernel0(const float *__restrict__ Data, */
/*                                  const float *__restrict__ FilterK, */
/*                                  const int Rank, */
/*                                  const int K0, */
/*                                  const float *__restrict__ Filter1, */
/*                                  const int K1, */
/*                                  const float *__restrict__ FilterR, */
/*                                  const int K2, */
/*                                  const float *__restrict__ FilterS, */
/*                                  const int K3, */
/*                                  float *__restrict__ Output) { */

/*   float            Output_local[2]; */
/*   __shared__ float pad_temp_shared[272]; */
/*   __shared__ float K0_shared[4]; */
/*   __shared__ float K1_shared[1]; */
/*   __shared__ float K2_shared[3]; */
/*   __shared__ float K3_shared[16]; */
/*   float            pad_temp_shared_local[6]; */
/*   float            K0_shared_local[1]; */
/*   float            K1_shared_local[1]; */
/*   float            K2_shared_local[3]; */
/*   float            K3_shared_local[1]; */

/*   for (int i = 0; i < 2; ++i) Output_local[i] = 0.0f; */
/* } */


Tensor cp4conv2d(Tensor const Input,
                 Tensor const FilterK,
                 Tensor const FilterC,
                 Tensor const FilterR,
                 Tensor const FilterS) {

  const int rank = FilterK.shape[1];
  const int FK   = FilterK.shape[0];
  const int FC   = FilterC.shape[0];
  const int FR   = FilterR.shape[0];
  const int FS   = FilterS.shape[0];
  const int N    = Input.shape[0];
  const int C    = Input.shape[1];
  const int H    = Input.shape[2];
  const int W    = Input.shape[3];

  // Zero Padding values
  const int ph   = (FR - 1) / 2;
  const int pw   = (FS - 1) / 2;
  const int padH = 2 * ph + H;
  const int padW = 2 * pw + W;

  Tensor Pad{ N, C, padH, padW };

  // clang-format off
  for (int n = 0; n < N; ++n)
  for (int c = 0; c < C; ++c)
  for (int h = 0; h < H; ++h)
  for (int w = 0; w < W; ++w)
    Pad[n*C*padH*padW + c*padH*padW + (h+ph)*padW + (w+pw)]
      = Input.m_data[n*C*H*W + c*H*W + (h)*W + (w)];

  Tensor Out{ N, C, H, W };

  for (int n = 0; n < N; ++n)
  for (int c = 0; c < C; ++c)
  for (int h = 0; h < H; ++h)
  for (int w = 0; w < W; ++w)
  for (int fk = 0; fk < FK; ++fk)
  for (int fc = 0; fc < FC; ++fc)
  for (int fr = 0; fr < FR; ++fr)
  for (int fs = 0; fs < FS; ++fs)
  for (int r = 0; r < rank; ++r)
    Out[n*C*H*W + c*H*W + h*W + w] += 
      Pad.m_data[n*C*padH*padW + c*padH*padW + (h+fr)*padW + (w+fs)]
      * FilterK.m_data[fk*rank + r]
      * FilterC.m_data[fc*rank + r]
      * FilterR.m_data[fr*rank + r]
      * FilterS.m_data[fs*rank + r];
  // clang-format on

  /* dim3   gridDim0(Input.shape[0], 1, 1); */
  /* dim3   blockDim0(Input.shape[1], Input.shape[2], Input.shape[3]); */
  /* cp4_base_kernel0<<<gridDim0, blockDim0>>>(Input.m_data, */
  /*                                           FilterK.m_data, */
  /*                                           rank, */
  /*                                           K0, */
  /*                                           FilterC.m_data, */
  /*                                           K1, */
  /*                                           FilterR.m_data, */
  /*                                           K2, */
  /*                                           FilterS.m_data, */
  /*                                           K3, */
  /*                                           Out.m_data); */

  return Out;
}
