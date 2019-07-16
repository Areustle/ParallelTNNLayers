#include "../Tensor.h"

__constant__ float carray[1 << 13];

__global__ void conv2d_full_kernel(const float *__restrict__ Input,
                                   const int C,
                                   const int K,
                                   const int R,
                                   const int FRCenter,
                                   const int S,
                                   const int FSCenter,
                                   float *__restrict__ Out) {

  /* __shared__ float shared_in [1024]; */

  // clang-format off
  for (int k = 0; k < K; ++k) {
    float sum = 0.0f;
    for (int c = 0; c < C; ++c)
    for (int r = 0; r < R; ++r)
    for (int s = 0; s < S; ++s) {

      /* const int   fidx = k * C * R * S + c * R * S + r * S + s; */
      /* const float fil  = carray[fidx]; */
      /* const int   hIdx = blockIdx.y + (r - FRCenter); */
      /* const int   wIdx = blockIdx.z + (s - FSCenter); */

      sum += 1;

      /* if (hIdx >= 0 && hIdx < gridDim.y && wIdx >= 0 && wIdx < gridDim.z) { */
      /*   sum += Input[blockIdx.x*C*gridDim.y*gridDim.z */
      /*                + c*gridDim.y*gridDim.z */ 
      /*                + hIdx*gridDim.z */ 
      /*                + wIdx] * fil; */
      /* } */
    }
    /* Out[blockIdx.x*C*gridDim.y*gridDim.z */
    /*     + k*gridDim.y*gridDim.z */
    /*     + gridDim.y*gridDim.z */
    /*     + blockIdx.z] = sum; */
  }
  // clang-format on
}

int main() {

  size_t PROFCOUNT = 10000;


  const int N = 1, C = 16, H = 32, W = 32;
  const int K = 16, R = 3, S = 3;

  const int FRCenter = R / 2;
  const int FSCenter = S / 2;

  /* float *In; */
  /* float *Out; */
  float  Fil[K * C * R * S];

  /* cudaMalloc(&In, (N * C * H * W) * sizeof(float)); */
  Tensor In = { N, C, H, W };
  cudaMemcpyToSymbol(carray, Fil, (K * C * R * S) * sizeof(float));
  Tensor Out = { N, C, H, W };

  dim3 gridDim0(N, H / 32, W / 32);
  dim3 blockDim0(1, 32, 32);

  for (size_t i = 0; i < PROFCOUNT; ++i) {
    conv2d_full_kernel<<<gridDim0, blockDim0>>>(
        In.m_data, C, K, R, FRCenter, S, FSCenter, Out.m_data);
    cudaDeviceSynchronize();
  }

  /* cudaFree(In); */
  cudaFree(Fil);
}
