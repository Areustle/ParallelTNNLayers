#if GOOGLE_CUDA
#define EIGEN_USE_GPU

__global__ void default_function_kernel0(const float* __restrict__ Data,
    const float* __restrict__ K0,
    const float* __restrict__ K1,
    const float* __restrict__ K2,
    float* __restrict__ Output) {

   float Output_local[1];
  __shared__ float Data_shared[128];
  __shared__ float K0_shared[32];
  __shared__ float K1_shared[64];
  __shared__ float K2_shared[1];
  Output_local[0] = 0.000000e+00f;
  for (int rr_outer = 0; rr_outer < 137; ++rr_outer) {
    for (int rs0_outer = 0; rs0_outer < 2; ++rs0_outer) {
      for (int rs2_outer = 0; rs2_outer < 16; ++rs2_outer) {
        __syncthreads();
        for (int ax0_ax1_ax2_ax3_fused_fused_fused_inner_inner_inner = 0; ax0_ax1_ax2_ax3_fused_fused_fused_inner_inner_inner < 8; ++ax0_ax1_ax2_ax3_fused_fused_fused_inner_inner_inner) {
          Data_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + ax0_ax1_ax2_ax3_fused_fused_fused_inner_inner_inner)] = Data[((((((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + ax0_ax1_ax2_ax3_fused_fused_fused_inner_inner_inner) / 128) * 4096) + (rs0_outer * 2048)) + (((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 8)) + ax0_ax1_ax2_ax3_fused_fused_fused_inner_inner_inner) % 128) * 16)) + rs2_outer)];
        }
        for (int ax0_ax1_ax2_fused_fused_inner_inner_inner = 0; ax0_ax1_ax2_fused_fused_inner_inner_inner < 2; ++ax0_ax1_ax2_fused_fused_inner_inner_inner) {
          K0_shared[(((((int)threadIdx.z) * 8) + (((int)threadIdx.y) * 2)) + ax0_ax1_ax2_fused_fused_inner_inner_inner)] = K0[(((((rs0_outer * 4384) + (((int)threadIdx.z) * 1096)) + (((int)threadIdx.y) * 274)) + (ax0_ax1_ax2_fused_fused_inner_inner_inner * 137)) + rr_outer)];
        }
        for (int ax0_ax1_ax2_fused_fused_inner_inner_inner1 = 0; ax0_ax1_ax2_fused_fused_inner_inner_inner1 < 4; ++ax0_ax1_ax2_fused_fused_inner_inner_inner1) {
          K1_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 4)) + ax0_ax1_ax2_fused_fused_inner_inner_inner1)] = K1[((((((int)threadIdx.z) * 2192) + (((int)threadIdx.y) * 548)) + (ax0_ax1_ax2_fused_fused_inner_inner_inner1 * 137)) + rr_outer)];
        }
        if (((int)threadIdx.y) < (1 - ((int)threadIdx.z))) {
          if (((int)threadIdx.y) < 1) {
            if (((int)threadIdx.y) < ((16 - rs2_outer) - ((int)threadIdx.z))) {
              K2_shared[(((int)threadIdx.y) + ((int)threadIdx.z))] = K2[(((((((int)threadIdx.y) * 548) + (((int)threadIdx.z) * 548)) + (rs2_outer * 548)) + (((int)blockIdx.x) * 137)) + rr_outer)];
            }
          }
        }
        __syncthreads();
        for (int rs0_inner = 0; rs0_inner < 8; ++rs0_inner) {
          for (int rs1_inner = 0; rs1_inner < 16; ++rs1_inner) {
            Output_local[0] = (Output_local[0] + (((Data_shared[((rs0_inner * 16) + rs1_inner)] * K0_shared[((rs0_inner * 4) + ((int)threadIdx.z))]) * K1_shared[((rs1_inner * 4) + ((int)threadIdx.y))]) * K2_shared[0]));
          }
        }
      }
    }
  }
  Output[(((((int)threadIdx.z) * 16) + (((int)threadIdx.y) * 4)) + ((int)blockIdx.x))] = Output_local[0];
}

void DenseCpKernelLauncher(const float* U, const float* K0,
    const float* K1, const float* KC, float* V){

  dim3 gridDim0(4, 1, 1);
  dim3 blockDim0(1, 4, 4);

  default_function_kernel0<<<gridDim0, blockDim0>>>(U, K0, K1, KC, V);
  cudaDeviceSynchronize();

}

#endif
