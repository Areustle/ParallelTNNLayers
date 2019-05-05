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

extern "C" __global__ void default_function_kernel1( const float* __restrict__ U0, const float* __restrict__ K1,  float* __restrict__ DepthwiseConv2d) {
   float DepthwiseConv2d_local[16];
  __shared__ float PaddedInput_shared[512];
  __shared__ float K1_shared[2];
  for (int b_c_init = 0; b_c_init < 2; ++b_c_init) {
    for (int i_c_init = 0; i_c_init < 4; ++i_c_init) {
      for (int c_c_init = 0; c_c_init < 2; ++c_c_init) {
        DepthwiseConv2d_local[(((b_c_init * 8) + (i_c_init * 2)) + c_c_init)] = 0.000000e+00f;
      }
    }
  }
  for (int di_outer = 0; di_outer < 3; ++di_outer) {
    for (int dj_outer = 0; dj_outer < 3; ++dj_outer) {
      __syncthreads();
      for (int ax0_ax1_ax2_ax3_fused_fused_fused_inner_inner_inner = 0; ax0_ax1_ax2_ax3_fused_fused_fused_inner_inner_inner < 16; ++ax0_ax1_ax2_ax3_fused_fused_fused_inner_inner_inner) {
        PaddedInput_shared[(((((int)threadIdx.z) * 256) + (((int)threadIdx.y) * 16)) + ax0_ax1_ax2_ax3_fused_fused_fused_inner_inner_inner)] = (((((((1 - ((((((int)threadIdx.y) * 16) + ax0_ax1_ax2_ax3_fused_fused_fused_inner_inner_inner) % 128) / 32)) - di_outer) <= ((((int)blockIdx.z) % 8) * 4)) && (((((int)blockIdx.z) % 8) * 4) < ((33 - ((((((int)threadIdx.y) * 16) + ax0_ax1_ax2_ax3_fused_fused_fused_inner_inner_inner) % 128) / 32)) - di_outer))) && (((1 - ((((((int)threadIdx.y) * 16) + ax0_ax1_ax2_ax3_fused_fused_fused_inner_inner_inner) % 32) / 2)) - dj_outer) <= (((int)blockIdx.y) * 16))) && ((((int)blockIdx.y) * 16) < ((33 - ((((((int)threadIdx.y) * 16) + ax0_ax1_ax2_ax3_fused_fused_fused_inner_inner_inner) % 32) / 2)) - dj_outer))) ? U0[(((((((((((((((int)blockIdx.z) / 8) * 24576) + (((int)threadIdx.z) * 12288)) + ((((((int)threadIdx.y) * 16) + ax0_ax1_ax2_ax3_fused_fused_fused_inner_inner_inner) / 128) * 6144)) + ((((int)blockIdx.z) % 8) * 768)) + (((((((int)threadIdx.y) * 16) + ax0_ax1_ax2_ax3_fused_fused_fused_inner_inner_inner) % 128) / 32) * 192)) + (di_outer * 192)) + (((int)blockIdx.y) * 96)) + (((((((int)threadIdx.y) * 16) + ax0_ax1_ax2_ax3_fused_fused_fused_inner_inner_inner) % 32) / 2) * 6)) + (dj_outer * 6)) + (((int)blockIdx.x) * 2)) + (ax0_ax1_ax2_ax3_fused_fused_fused_inner_inner_inner % 2)) - 198)] : 0.000000e+00f);
      }
      if (((int)threadIdx.y) < (2 - ((int)threadIdx.z))) {
        if (((int)threadIdx.y) < 1) {
          if (((((int)threadIdx.y) + ((int)threadIdx.z)) / 2) < (3 - di_outer)) {
            K1_shared[(((int)threadIdx.y) + ((int)threadIdx.z))] = K1[(((((((((int)threadIdx.y) + ((int)threadIdx.z)) / 2) * 18) + (di_outer * 18)) + (dj_outer * 6)) + (((int)blockIdx.x) * 2)) + ((((int)threadIdx.y) + ((int)threadIdx.z)) % 2))];
          }
        }
      }
      __syncthreads();
      for (int b_c = 0; b_c < 2; ++b_c) {
        for (int i_c = 0; i_c < 4; ++i_c) {
          for (int c_c = 0; c_c < 2; ++c_c) {
            DepthwiseConv2d_local[(((b_c * 8) + (i_c * 2)) + c_c)] = (DepthwiseConv2d_local[(((b_c * 8) + (i_c * 2)) + c_c)] + (PaddedInput_shared[(((((((int)threadIdx.z) * 256) + (b_c * 128)) + (i_c * 32)) + (((int)threadIdx.y) * 2)) + c_c)] * K1_shared[c_c]));
          }
        }
      }
    }
  }
  for (int b_inner_inner_inner = 0; b_inner_inner_inner < 2; ++b_inner_inner_inner) {
    for (int i_inner_inner_inner = 0; i_inner_inner_inner < 4; ++i_inner_inner_inner) {
      for (int c_inner_inner_inner = 0; c_inner_inner_inner < 2; ++c_inner_inner_inner) {
        DepthwiseConv2d[((((((((((((int)blockIdx.z) / 8) * 24576) + (((int)threadIdx.z) * 12288)) + (b_inner_inner_inner * 6144)) + ((((int)blockIdx.z) % 8) * 768)) + (i_inner_inner_inner * 192)) + (((int)blockIdx.y) * 96)) + (((int)threadIdx.y) * 6)) + (((int)blockIdx.x) * 2)) + c_inner_inner_inner)] = DepthwiseConv2d_local[(((b_inner_inner_inner * 8) + (i_inner_inner_inner * 2)) + c_inner_inner_inner)];
      }
    }
  }
}


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

  dim3 gridDim0(1, 2, 32);
  dim3 blockDim0(1, 16, 2);

  dim3 gridDim1(3, 2, 16);
  dim3 blockDim1(1, 16, 2);

  dim3 gridDim2(1, 8, 16);
  dim3 blockDim2(8, 1, 16);

  default_function_kernel0<<<gridDim0, blockDim0>>>(U, K0, U0);
  default_function_kernel1<<<gridDim1, blockDim1>>>(U0, K1, U1);
  default_function_kernel2<<<gridDim2, blockDim2>>>(U1, K2, V);
  cudaDeviceSynchronize();

}

#endif
