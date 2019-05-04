#if GOOGLE_CUDA
#define EIGEN_USE_GPU

extern "C" __global__ void default_function_kernel0( float* __restrict__ U,  float* __restrict__ K0,  float* __restrict__ U0) {
   float U0_local[2];
  __shared__ float U_shared[64];
  __shared__ float K0_shared[24];
  for (int n_inner_outer = 0; n_inner_outer < 2; ++n_inner_outer) {
    for (int r_c_init = 0; r_c_init < 2; ++r_c_init) {
      U0_local[r_c_init] = 0.000000e+00f;
    }
    for (int k0_c_outer = 0; k0_c_outer < 4; ++k0_c_outer) {
      __syncthreads();
      for (int ax0_ax1_ax2_ax3_fused_fused_fused_inner_inner_inner = 0; ax0_ax1_ax2_ax3_fused_fused_fused_inner_inner_inner < 2; ++ax0_ax1_ax2_ax3_fused_fused_fused_inner_inner_inner) {
        if (((((((int)threadIdx.y) * 4) + (((int)threadIdx.x) * 2)) + ax0_ax1_ax2_ax3_fused_fused_fused_inner_inner_inner) / 16) < (4 - ((int)threadIdx.z))) {
          if ((((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 4)) + (((int)threadIdx.x) * 2)) < (64 - ax0_ax1_ax2_ax3_fused_fused_fused_inner_inner_inner)) {
            if (((((int)threadIdx.y) * 4) + (((int)threadIdx.x) * 2)) < (16 - ax0_ax1_ax2_ax3_fused_fused_fused_inner_inner_inner)) {
              if ((((int)threadIdx.x) * 2) < (4 - ax0_ax1_ax2_ax3_fused_fused_fused_inner_inner_inner)) {
                if ((n_inner_outer * 4) < ((8 - ((((((int)threadIdx.y) * 4) + (((int)threadIdx.x) * 2)) + ax0_ax1_ax2_ax3_fused_fused_fused_inner_inner_inner) / 16)) - ((int)threadIdx.z))) {
                  U_shared[((((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 4)) + (((int)threadIdx.x) * 2)) + ax0_ax1_ax2_ax3_fused_fused_fused_inner_inner_inner)] = U[((((((((n_inner_outer * 65536) + (((((((int)threadIdx.y) * 4) + (((int)threadIdx.x) * 2)) + ax0_ax1_ax2_ax3_fused_fused_fused_inner_inner_inner) / 16) * 16384)) + (((int)threadIdx.z) * 16384)) + (((int)blockIdx.z) * 512)) + (((int)blockIdx.y) * 64)) + ((((((((int)threadIdx.y) * 4) + (((int)threadIdx.x) * 2)) + ax0_ax1_ax2_ax3_fused_fused_fused_inner_inner_inner) % 16) / 4) * 16)) + (k0_c_outer * 4)) + (((((int)threadIdx.x) * 2) + ax0_ax1_ax2_ax3_fused_fused_fused_inner_inner_inner) % 4))];
                }
              }
            }
          }
        }
      }
      if ((((((int)threadIdx.y) * 2) + ((int)threadIdx.x)) / 6) < (4 - ((int)threadIdx.z))) {
        if (((((int)threadIdx.z) * 6) + (((int)threadIdx.y) * 2)) < (24 - ((int)threadIdx.x))) {
          if ((((int)threadIdx.y) * 2) < (6 - ((int)threadIdx.x))) {
            if (((int)threadIdx.x) < 2) {
              if ((k0_c_outer * 4) < ((16 - (((((int)threadIdx.y) * 2) + ((int)threadIdx.x)) / 6)) - ((int)threadIdx.z))) {
                K0_shared[(((((int)threadIdx.z) * 6) + (((int)threadIdx.y) * 2)) + ((int)threadIdx.x))] = K0[((((k0_c_outer * 24) + (((int)threadIdx.z) * 6)) + (((int)threadIdx.y) * 2)) + ((int)threadIdx.x))];
              }
            }
          }
        }
      }
      __syncthreads();
      for (int k0_c_inner = 0; k0_c_inner < 4; ++k0_c_inner) {
        for (int r_c = 0; r_c < 2; ++r_c) {
          U0_local[r_c] = (U0_local[r_c] + (U_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 4)) + k0_c_inner)] * K0_shared[(((k0_c_inner * 6) + (((int)threadIdx.x) * 2)) + r_c)]));
        }
      }
    }
    for (int r_inner_inner_inner = 0; r_inner_inner_inner < 2; ++r_inner_inner_inner) {
      U0[(((((((n_inner_outer * 24576) + (((int)threadIdx.z) * 6144)) + (((int)blockIdx.z) * 192)) + (((int)blockIdx.y) * 24)) + (((int)threadIdx.y) * 6)) + (((int)threadIdx.x) * 2)) + r_inner_inner_inner)] = U0_local[r_inner_inner_inner];
    }
  }
}

/* void NMode32KernelLauncher(const float* U, const int I, const int J, const int S, */
/*                    const float* B, const int R, */
/*                    float* C) { */

void CpForwardUnfusedKernelLauncher(const float* U, const float* K0, float* U0){
  default_function_kernel0<<<1, 256>>>(U, K0, U0);
  cudaDeviceSynchronize();
}

#endif
