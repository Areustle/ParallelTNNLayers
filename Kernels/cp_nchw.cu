#if GOOGLE_CUDA
#define EIGEN_USE_GPU


extern "C" __global__ void default_function_kernel0( float* __restrict__ Data,  float* __restrict__ Kernel,  float* __restrict__ Output) {
   float Output_local[1];
  __shared__ float Data_shared[512];
  __shared__ float Kernel_shared[1];
  #pragma unroll
  for (int w_inner_outer = 0; w_inner_outer < 2; ++w_inner_outer) {
    Output_local[0] = 0.000000e+00f;
    #pragma unroll
    for (int k0_c_outer = 0; k0_c_outer < 16; ++k0_c_outer) {
      __syncthreads();
      Data_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 16)) + ((int)threadIdx.x))] = Data[(((((((((((int)blockIdx.z) / 6) * 65536) + ((((((int)threadIdx.y) * 16) + ((int)threadIdx.x)) / 128) * 16384)) + (((int)threadIdx.z) * 16384)) + (k0_c_outer * 1024)) + (((int)blockIdx.y) * 256)) + (((((((int)threadIdx.y) * 16) + ((int)threadIdx.x)) % 128) / 16) * 32)) + (w_inner_outer * 16)) + ((int)threadIdx.x))];
      if (((int)threadIdx.x) < ((1 - ((int)threadIdx.z)) - ((int)threadIdx.y))) {
        if (((int)threadIdx.x) < (1 - ((int)threadIdx.y))) {
          if (((int)threadIdx.x) < 1) {
            if (((int)threadIdx.x) < (((16 - k0_c_outer) - ((int)threadIdx.z)) - ((int)threadIdx.y))) {
              Kernel_shared[((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z))] = Kernel[(((((((int)threadIdx.x) * 6) + (((int)threadIdx.y) * 6)) + (((int)threadIdx.z) * 6)) + (k0_c_outer * 6)) + (((int)blockIdx.z) % 6))];
            }
          }
        }
      }
      __syncthreads();
      Output_local[0] = (Output_local[0] + (Data_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 16)) + ((int)threadIdx.x))] * Kernel_shared[0]));
    }
    Output[((((((((((int)blockIdx.z) / 6) * 24576) + (((int)threadIdx.z) * 6144)) + ((((int)blockIdx.z) % 6) * 1024)) + (((int)blockIdx.y) * 256)) + (((int)threadIdx.y) * 32)) + (w_inner_outer * 16)) + ((int)threadIdx.x))] = Output_local[0];
  }
}


void Cp0NchwKernelLauncher(const float* Data, const float* Kernel, float* Output){

  dim3 gridDim(1, 4, 12);
  dim3 blockDim(16, 8, 4);

  default_function_kernel0<<<gridDim, blockDim>>>(Data, Kernel, Output);

  cudaDeviceSynchronize();
}


extern "C" __global__ void default_function_kernel1( float* __restrict__ Data,  float* __restrict__ Kernel,  float* __restrict__ Output) {
   float Output_local[2];
  __shared__ float PaddedInput_shared[864];
  __shared__ float Kernel_shared[9];
  for (int i_inner_outer = 0; i_inner_outer < 4; ++i_inner_outer) {
    for (int i_c_init = 0; i_c_init < 2; ++i_c_init) {
      Output_local[i_c_init] = 0.000000e+00f;
    }
    for (int di_outer = 0; di_outer < 3; ++di_outer) {
      __syncthreads();
      for (int ax0_ax1_ax2_ax3_fused_fused_fused_inner_inner_inner = 0; ax0_ax1_ax2_ax3_fused_fused_fused_inner_inner_inner < 3; ++ax0_ax1_ax2_ax3_fused_fused_fused_inner_inner_inner) {
        if (((((int)threadIdx.z) * 36) + (((int)threadIdx.x) * 3)) < (864 - ax0_ax1_ax2_ax3_fused_fused_fused_inner_inner_inner)) {
          if ((((int)threadIdx.x) * 3) < (36 - ax0_ax1_ax2_ax3_fused_fused_fused_inner_inner_inner)) {
            PaddedInput_shared[(((((int)threadIdx.z) * 36) + (((int)threadIdx.x) * 3)) + ax0_ax1_ax2_ax3_fused_fused_fused_inner_inner_inner)] = (((((((1 - ((((((int)threadIdx.x) * 3) + ax0_ax1_ax2_ax3_fused_fused_fused_inner_inner_inner) % 36) / 18)) - di_outer) <= ((((int)blockIdx.y) * 8) + (i_inner_outer * 2))) && (((((int)blockIdx.y) * 8) + (i_inner_outer * 2)) < ((33 - ((((((int)threadIdx.x) * 3) + ax0_ax1_ax2_ax3_fused_fused_fused_inner_inner_inner) % 36) / 18)) - di_outer))) && ((1 - (((((int)threadIdx.x) * 3) + ax0_ax1_ax2_ax3_fused_fused_fused_inner_inner_inner) % 18)) <= (((int)blockIdx.x) * 16))) && ((((int)blockIdx.x) * 16) < (33 - (((((int)threadIdx.x) * 3) + ax0_ax1_ax2_ax3_fused_fused_fused_inner_inner_inner) % 18)))) ? Data[((((((((((((((((int)threadIdx.z) * 36) + (((int)threadIdx.x) * 3)) + ax0_ax1_ax2_ax3_fused_fused_fused_inner_inner_inner) / 108) * 6144) + (((int)blockIdx.z) * 3072)) + ((((((((int)threadIdx.z) * 36) + (((int)threadIdx.x) * 3)) + ax0_ax1_ax2_ax3_fused_fused_fused_inner_inner_inner) % 108) / 36) * 1024)) + (((int)blockIdx.y) * 256)) + (i_inner_outer * 64)) + (((((((int)threadIdx.x) * 3) + ax0_ax1_ax2_ax3_fused_fused_fused_inner_inner_inner) % 36) / 18) * 32)) + (di_outer * 32)) + (((int)blockIdx.x) * 16)) + (((((int)threadIdx.x) * 3) + ax0_ax1_ax2_ax3_fused_fused_fused_inner_inner_inner) % 18)) - 33)] : 0.000000e+00f);
          }
        }
      }
      if (((int)threadIdx.x) < (9 - ((int)threadIdx.z))) {
        if (((int)threadIdx.x) < 1) {
          if (((((int)threadIdx.x) + ((int)threadIdx.z)) / 9) < (3 - di_outer)) {
            Kernel_shared[(((int)threadIdx.x) + ((int)threadIdx.z))] = Kernel[((((di_outer * 18) + (((((int)threadIdx.x) + ((int)threadIdx.z)) / 3) * 6)) + (((int)blockIdx.z) * 3)) + ((((int)threadIdx.x) + ((int)threadIdx.z)) % 3))];
          }
        }
      }
      __syncthreads();
      for (int dj_inner = 0; dj_inner < 3; ++dj_inner) {
        for (int i_c = 0; i_c < 2; ++i_c) {
          Output_local[i_c] = (Output_local[i_c] + (PaddedInput_shared[((((((int)threadIdx.z) * 36) + (i_c * 18)) + ((int)threadIdx.x)) + dj_inner)] * Kernel_shared[((dj_inner * 3) + (((int)threadIdx.z) % 3))]));
        }
      }
    }
    for (int i_inner_inner_inner = 0; i_inner_inner_inner < 2; ++i_inner_inner_inner) {
      Output[(((((((((((int)threadIdx.z) / 3) * 6144) + (((int)blockIdx.z) * 3072)) + ((((int)threadIdx.z) % 3) * 1024)) + (((int)blockIdx.y) * 256)) + (i_inner_outer * 64)) + (i_inner_inner_inner * 32)) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x))] = Output_local[i_inner_inner_inner];
    }
  }
}


void Cp1NchwKernelLauncher(const float* U, const float* K, float* V){

  dim3 gridDim(2, 4, 2);
  dim3 blockDim(16, 1, 24);
  default_function_kernel1<<<gridDim, blockDim>>>(U, K, V);
  cudaDeviceSynchronize();
}

extern "C" __global__ void default_function_kernel2( float* __restrict__ Data,  float* __restrict__ Kernel,  float* __restrict__ Output) {
   float Output_local[2];
  __shared__ float Data_shared[512];
  __shared__ float Kernel_shared[4];
  for (int n_inner_outer = 0; n_inner_outer < 2; ++n_inner_outer) {
    for (int r_inner_outer = 0; r_inner_outer < 2; ++r_inner_outer) {
      for (int w_inner_outer = 0; w_inner_outer < 2; ++w_inner_outer) {
        for (int r_c_init = 0; r_c_init < 2; ++r_c_init) {
          Output_local[r_c_init] = 0.000000e+00f;
        }
        for (int k0_c_outer = 0; k0_c_outer < 3; ++k0_c_outer) {
          __syncthreads();
          for (int ax0_ax1_ax2_ax3_fused_fused_fused_inner_inner_inner = 0; ax0_ax1_ax2_ax3_fused_fused_fused_inner_inner_inner < 2; ++ax0_ax1_ax2_ax3_fused_fused_fused_inner_inner_inner) {
            Data_shared[(((((int)threadIdx.y) * 32) + (((int)threadIdx.x) * 2)) + ax0_ax1_ax2_ax3_fused_fused_fused_inner_inner_inner)] = Data[((((((((((((int)blockIdx.z) / 4) * 12288) + (((((((int)threadIdx.y) * 32) + (((int)threadIdx.x) * 2)) + ax0_ax1_ax2_ax3_fused_fused_fused_inner_inner_inner) / 512) * 6144)) + (n_inner_outer * 6144)) + (k0_c_outer * 2048)) + ((((((((int)threadIdx.y) * 32) + (((int)threadIdx.x) * 2)) + ax0_ax1_ax2_ax3_fused_fused_fused_inner_inner_inner) % 512) / 256) * 1024)) + (((int)blockIdx.y) * 512)) + ((((((((int)threadIdx.y) * 32) + (((int)threadIdx.x) * 2)) + ax0_ax1_ax2_ax3_fused_fused_fused_inner_inner_inner) % 256) / 16) * 32)) + (w_inner_outer * 16)) + (((((int)threadIdx.x) * 2) + ax0_ax1_ax2_ax3_fused_fused_fused_inner_inner_inner) % 16))];
          }
          if (((int)threadIdx.x) < (4 - ((int)threadIdx.y))) {
            if (((int)threadIdx.x) < 1) {
              if ((k0_c_outer * 2) < (6 - ((((int)threadIdx.x) + ((int)threadIdx.y)) / 2))) {
                Kernel_shared[(((int)threadIdx.x) + ((int)threadIdx.y))] = Kernel[(((((k0_c_outer * 32) + (((((int)threadIdx.x) + ((int)threadIdx.y)) / 2) * 16)) + ((((int)blockIdx.z) % 4) * 4)) + (r_inner_outer * 2)) + ((((int)threadIdx.x) + ((int)threadIdx.y)) % 2))];
              }
            }
          }
          __syncthreads();
          for (int k0_c_inner = 0; k0_c_inner < 2; ++k0_c_inner) {
            for (int r_c = 0; r_c < 2; ++r_c) {
              Output_local[r_c] = (Output_local[r_c] + (Data_shared[(((k0_c_inner * 256) + (((int)threadIdx.y) * 16)) + ((int)threadIdx.x))] * Kernel_shared[((k0_c_inner * 2) + r_c)]));
            }
          }
        }
        for (int r_inner_inner_inner = 0; r_inner_inner_inner < 2; ++r_inner_inner_inner) {
          Output[((((((((((((int)blockIdx.z) / 4) * 32768) + (n_inner_outer * 16384)) + ((((int)blockIdx.z) % 4) * 4096)) + (r_inner_outer * 2048)) + (r_inner_inner_inner * 1024)) + (((int)blockIdx.y) * 512)) + (((int)threadIdx.y) * 32)) + (w_inner_outer * 16)) + ((int)threadIdx.x))] = Output_local[r_inner_inner_inner];
        }
      }
    }
  }
}



void Cp2NchwKernelLauncher(const float* U, const float* K, float* V){

  dim3 gridDim2(1, 2, 16);
  dim3 blockDim2(16, 16, 1);
  default_function_kernel2<<<gridDim2, blockDim2>>>(U, K, V);
  cudaDeviceSynchronize();

}

#endif
