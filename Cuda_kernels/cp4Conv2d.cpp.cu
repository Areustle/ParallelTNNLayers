#include "cp4Conv2d.h"

__global__ void cp4_base_kernel0(const float *__restrict__ Data,
                                 const float *__restrict__ Kernel0,
                                 const int Rank,
                                 const int K0,
                                 const float *__restrict__ Kernel1,
                                 const int K1,
                                 const float *__restrict__ Kernel2,
                                 const int K2,
                                 const float *__restrict__ Kernel3,
                                 const int K3,
                                 float *__restrict__ Output) {

  float            Output_local[2];
  __shared__ float pad_temp_shared[272];
  __shared__ float K0_shared[4];
  __shared__ float K1_shared[1];
  __shared__ float K2_shared[3];
  __shared__ float K3_shared[16];
  float            pad_temp_shared_local[6];
  float            K0_shared_local[1];
  float            K1_shared_local[1];
  float            K2_shared_local[3];
  float            K3_shared_local[1];

  for (int i = 0; i < 2; ++i) Output_local[i] = 0.0f;
}


Tensor cp4conv2d(Tensor const Input,
                 Tensor const Kernel0,
                 Tensor const Kernel1,
                 Tensor const Kernel2,
                 Tensor const Kernel3) {

  const size_t rank = Kernel0.shape[1];
  const int    K0   = Kernel0.shape[0];
  const int    K1   = Kernel1.shape[0];
  const int    K2   = Kernel2.shape[0];
  const int    K3   = Kernel3.shape[0];
  Tensor Out{ Input.shape[0], Input.shape[1], Input.shape[2], Input.shape[3] };
  dim3   gridDim0(Input.shape[0], 1, 1);
  dim3   blockDim0(Input.shape[1], Input.shape[2], Input.shape[3]);

  cp4_base_kernel0<<<gridDim0, blockDim0>>>(Input.m_data,
                                            Kernel0.m_data,
                                            rank,
                                            K0,
                                            Kernel1.m_data,
                                            K1,
                                            Kernel2.m_data,
                                            K2,
                                            Kernel3.m_data,
                                            K3,
                                            Out.m_data);

  return Out;
}
