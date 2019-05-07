#if GOOGLE_CUDA
#define EIGEN_USE_GPU

__global__ void default_function_kernel0(const float* __restrict__ U, const float* __restrict__ K0,  float* __restrict__ U0) {
   float U0_local[2];
  __shared__ float U_shared[512];
  __shared__ float K0_shared[2];
  #pragma unroll
  for (int n_inner_outer = 0; n_inner_outer < 2; ++n_inner_outer) {
    #pragma unroll
    for (int w_c_init = 0; w_c_init < 2; ++w_c_init) {
      U0_local[w_c_init] = 0.000000e+00f;
    }
    #pragma unroll
    for (int k0_c_outer = 0; k0_c_outer < 16; ++k0_c_outer) {
      __syncthreads();
      U_shared[(((((int)threadIdx.z) * 8) + (((int)threadIdx.y) * 2)) + ((int)threadIdx.x))] = U[(((((((n_inner_outer * 65536) + (((((((int)threadIdx.z) * 8) + (((int)threadIdx.y) * 2)) + ((int)threadIdx.x)) / 128) * 16384)) + (k0_c_outer * 1024)) + (((int)blockIdx.z) * 512)) + ((((((((int)threadIdx.z) * 8) + (((int)threadIdx.y) * 2)) + ((int)threadIdx.x)) % 128) / 8) * 32)) + (((int)blockIdx.y) * 8)) + ((((int)threadIdx.y) * 2) + ((int)threadIdx.x)))];
      if (((int)threadIdx.x) < ((2 - ((int)threadIdx.z)) - ((int)threadIdx.y))) {
        if (((int)threadIdx.x) < (1 - ((int)threadIdx.y))) {
          if (((int)threadIdx.x) < 1) {
            if ((((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) / 2) < (16 - k0_c_outer)) {
              K0_shared[((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z))] = K0[(((((((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) / 2) * 6) + (k0_c_outer * 6)) + (((int)blockIdx.x) * 2)) + (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) % 2))];
            }
          }
        }
      }
      __syncthreads();
      #pragma unroll
      for (int w_c = 0; w_c < 2; ++w_c) {
        U0_local[w_c] = (U0_local[w_c] + (U_shared[(((((int)threadIdx.z) * 8) + (((int)threadIdx.y) * 2)) + w_c)] * K0_shared[((int)threadIdx.x)]));
      }
    }
    #pragma unroll
    for (int w_inner_inner_inner = 0; w_inner_inner_inner < 2; ++w_inner_inner_inner) {
      U0[(((((((((n_inner_outer * 24576) + ((((int)threadIdx.z) / 16) * 6144)) + (((int)blockIdx.z) * 3072)) + ((((int)threadIdx.z) % 16) * 192)) + (((int)blockIdx.y) * 48)) + (((int)threadIdx.y) * 12)) + (w_inner_inner_inner * 6)) + (((int)blockIdx.x) * 2)) + ((int)threadIdx.x))] = U0_local[w_inner_inner_inner];
    }
  }
}


extern "C" __global__ void default_function_kernel1( const float* __restrict__ U0, const float* __restrict__ K1,  float* __restrict__ DepthwiseConv2d) {


extern "C" __global__ void default_function_kernel2(const float* __restrict__ U1, const float* __restrict__ K2,  float* __restrict__ V) {
   float V_local[8];
  __shared__ float U1_shared[192];
  __shared__ float K2_shared[48];
  for (int w_c_init = 0; w_c_init < 4; ++w_c_init) {
    for (int r_c_init = 0; r_c_init < 2; ++r_c_init) {
      V_local[((w_c_init * 2) + r_c_init)] = 0.000000e+00f;
    }
  }
  for (int k0_c_outer = 0; k0_c_outer < 2; ++k0_c_outer) {
    __syncthreads();
    for (int ax0_ax1_ax2_ax3_fused_fused_fused_inner_inner_inner = 0; ax0_ax1_ax2_ax3_fused_fused_fused_inner_inner_inner < 2; ++ax0_ax1_ax2_ax3_fused_fused_fused_inner_inner_inner) {
      if (((((int)threadIdx.z) * 12) + (((int)threadIdx.x) * 2)) < (192 - ax0_ax1_ax2_ax3_fused_fused_fused_inner_inner_inner)) {
        if ((((int)threadIdx.x) * 2) < (12 - ax0_ax1_ax2_ax3_fused_fused_fused_inner_inner_inner)) {
          if (((((int)blockIdx.z) / 8) * 4) < (8 - ((((((int)threadIdx.z) * 12) + (((int)threadIdx.x) * 2)) + ax0_ax1_ax2_ax3_fused_fused_fused_inner_inner_inner) / 48))) {
            U1_shared[(((((int)threadIdx.z) * 12) + (((int)threadIdx.x) * 2)) + ax0_ax1_ax2_ax3_fused_fused_fused_inner_inner_inner)] = U1[(((((((((((int)blockIdx.z) / 8) * 24576) + (((((((int)threadIdx.z) * 12) + (((int)threadIdx.x) * 2)) + ax0_ax1_ax2_ax3_fused_fused_fused_inner_inner_inner) / 48) * 6144)) + ((((int)blockIdx.z) % 8) * 768)) + ((((((((int)threadIdx.z) * 12) + (((int)threadIdx.x) * 2)) + ax0_ax1_ax2_ax3_fused_fused_fused_inner_inner_inner) % 48) / 12) * 192)) + (((int)blockIdx.y) * 24)) + (((((((int)threadIdx.x) * 2) + ax0_ax1_ax2_ax3_fused_fused_fused_inner_inner_inner) % 12) / 3) * 6)) + (k0_c_outer * 3)) + (((((int)threadIdx.x) * 2) + ax0_ax1_ax2_ax3_fused_fused_fused_inner_inner_inner) % 3))];
          }
        }
      }
    }
    if ((((int)threadIdx.z) * 3) < (48 - ((int)threadIdx.x))) {
      if (((int)threadIdx.x) < 3) {
        if ((k0_c_outer * 3) < (6 - (((((int)threadIdx.z) * 3) + ((int)threadIdx.x)) / 16))) {
          K2_shared[((((int)threadIdx.z) * 3) + ((int)threadIdx.x))] = K2[(((k0_c_outer * 48) + (((int)threadIdx.z) * 3)) + ((int)threadIdx.x))];
        }
      }
    }
    __syncthreads();
    for (int k0_c_inner = 0; k0_c_inner < 3; ++k0_c_inner) {
      for (int w_c = 0; w_c < 4; ++w_c) {
        for (int r_c = 0; r_c < 2; ++r_c) {
          V_local[((w_c * 2) + r_c)] = (V_local[((w_c * 2) + r_c)] + (U1_shared[(((((int)threadIdx.z) * 12) + (w_c * 3)) + k0_c_inner)] * K2_shared[(((k0_c_inner * 16) + (((int)threadIdx.x) * 2)) + r_c)]));
        }
      }
    }
  }
  for (int w_inner_inner_inner = 0; w_inner_inner_inner < 4; ++w_inner_inner_inner) {
    for (int r_inner_inner_inner = 0; r_inner_inner_inner < 2; ++r_inner_inner_inner) {
      V[(((((((((((int)blockIdx.z) / 8) * 65536) + ((((int)threadIdx.z) / 4) * 16384)) + ((((int)blockIdx.z) % 8) * 2048)) + ((((int)threadIdx.z) % 4) * 512)) + (((int)blockIdx.y) * 64)) + (w_inner_inner_inner * 16)) + (((int)threadIdx.x) * 2)) + r_inner_inner_inner)] = V_local[((w_inner_inner_inner * 2) + r_inner_inner_inner)];
    }
  }
}




void CpForwardUnfusedKernelLauncher(const float* U,
    const float* K0, const float* K1, const float* K2,
    float* U0, float* U1, float* V){

  dim3 gridDim0(3, 4, 2);
  dim3 blockDim0(2, 4, 64);
  default_function_kernel0<<<gridDim0, blockDim0>>>(U, K0, U0);
  cudaDeviceSynchronize();

  /* dim3 gridDim1(3, 2, 16); */
  /* dim3 blockDim1(1, 16, 2); */
  default_function_kernel1<<<gridDim1, blockDim1>>>(U0, K1, U1);
  cudaDeviceSynchronize();

  /* dim3 gridDim2(1, 8, 16); */
  /* dim3 blockDim2(8, 1, 16); */
  default_function_kernel2<<<gridDim2, blockDim2>>>(U1, K2, V);
  cudaDeviceSynchronize();

}

#endif
