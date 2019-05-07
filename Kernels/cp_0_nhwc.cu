#if GOOGLE_CUDA
#define EIGEN_USE_GPU

__global__ void default_function_kernel0(const float* __restrict__ U, const float* __restrict__ K0,  float* __restrict__ U0) {
  float U0_local[1];
  __shared__ float U_shared[128];
  __shared__ float K0_shared[4];
  for (int n_inner_outer = 0; n_inner_outer < 2; ++n_inner_outer) {
    for (int h_inner_outer = 0; h_inner_outer < 2; ++h_inner_outer) {
      for (int r_inner_outer = 0; r_inner_outer < 6; ++r_inner_outer) {
        U0_local[0] = 0.000000e+00f;
        for (int k0_c_outer = 0; k0_c_outer < 4; ++k0_c_outer) {
          __syncthreads();
          for (int ax0_ax1_ax2_ax3_fused_fused_fused_inner_inner_inner = 0; ax0_ax1_ax2_ax3_fused_fused_fused_inner_inner_inner < 4; ++ax0_ax1_ax2_ax3_fused_fused_fused_inner_inner_inner) {
            U_shared[(((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 4)) + ax0_ax1_ax2_ax3_fused_fused_fused_inner_inner_inner)] = U[(((((((((((((int)blockIdx.z) / 8) * 32768) + (((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 4)) + ax0_ax1_ax2_ax3_fused_fused_fused_inner_inner_inner) / 128) * 16384)) + (n_inner_outer * 16384)) + ((((int)blockIdx.z) % 8) * 2048)) + (h_inner_outer * 1024)) + ((((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 4)) + ax0_ax1_ax2_ax3_fused_fused_fused_inner_inner_inner) % 128) / 64) * 512)) + (((int)blockIdx.y) * 256)) + ((((((int)threadIdx.y) * 4) + ax0_ax1_ax2_ax3_fused_fused_fused_inner_inner_inner) / 4) * 16)) + (k0_c_outer * 4)) + ax0_ax1_ax2_ax3_fused_fused_fused_inner_inner_inner)];
          }
          if ((((int)threadIdx.z) * 2) < (4 - ((int)threadIdx.y))) {
            if (((int)threadIdx.y) < 2) {
              if (((k0_c_outer * 4) + (((int)threadIdx.z) * 2)) < (16 - ((int)threadIdx.y))) {
                K0_shared[((((int)threadIdx.z) * 2) + ((int)threadIdx.y))] = K0[((((k0_c_outer * 24) + (((int)threadIdx.z) * 12)) + (((int)threadIdx.y) * 6)) + r_inner_outer)];
              }
            }
          }
          __syncthreads();
          for (int k0_c_inner = 0; k0_c_inner < 4; ++k0_c_inner) {
            U0_local[0] = (U0_local[0] + (U_shared[(((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 4)) + k0_c_inner)] * K0_shared[k0_c_inner]));
          }
        }
        U0[(((((((((((int)blockIdx.z) / 8) * 12288) + (n_inner_outer * 6144)) + ((((int)blockIdx.z) % 8) * 768)) + (h_inner_outer * 384)) + (((int)threadIdx.z) * 192)) + (((int)blockIdx.y) * 96)) + (((int)threadIdx.y) * 6)) + r_inner_outer)] = U0_local[0];
      }
    }
  }
}


void Cp0NhwcKernelLauncher(const float* U, const float* K0, float* U0){

  dim3 gridDim0(1, 2, 32);
  dim3 blockDim0(1, 16, 2);

  default_function_kernel0<<<gridDim0, blockDim0>>>(U, K0, U0);
  cudaDeviceSynchronize();

}

#endif
