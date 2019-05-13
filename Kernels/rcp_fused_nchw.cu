#if GOOGLE_CUDA
#define EIGEN_USE_GPU

__global__ void default_function_kernel0(const float* __restrict__ Data,
    const float* __restrict__ K0,
    const float* __restrict__ K1,
    const float* __restrict__ KC,
    float* __restrict__ Output) {

     float Output_local[4];
  __shared__ float pad_temp_shared[512];
  __shared__ float K0_shared[8];
  __shared__ float K1_shared[2];
  __shared__ float KC_shared[1];
  for (int ww_inner_outer = 0; ww_inner_outer < 2; ++ww_inner_outer) {
    #pragma unroll
    for (int tt0_c_init = 0; tt0_c_init < 2; ++tt0_c_init) {
      #pragma unroll
      for (int tt1_c_init = 0; tt1_c_init < 2; ++tt1_c_init) {
        Output_local[((tt0_c_init * 2) + tt1_c_init)] = 0.000000e+00f;
      }
    }
    for (int rr_outer = 0; rr_outer < 11; ++rr_outer) {
      #pragma unroll
      for (int rs1_outer = 0; rs1_outer < 4; ++rs1_outer) {
        #pragma unroll
        for (int rh_outer = 0; rh_outer < 3; ++rh_outer) {
          #pragma unroll
          for (int rw_outer = 0; rw_outer < 3; ++rw_outer) {
            __syncthreads();
            #pragma unroll
            for (int ax0_ax1_ax2_ax3_ax4_fused_fused_fused_fused_inner_inner_inner = 0; ax0_ax1_ax2_ax3_ax4_fused_fused_fused_fused_inner_inner_inner < 4; ++ax0_ax1_ax2_ax3_ax4_fused_fused_fused_fused_inner_inner_inner) {
              pad_temp_shared[(((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 4)) + ax0_ax1_ax2_ax3_ax4_fused_fused_fused_fused_inner_inner_inner)] = (((((((1 - (((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 4)) + ax0_ax1_ax2_ax3_ax4_fused_fused_fused_fused_inner_inner_inner) % 128) / 16)) - rh_outer) <= (((int)blockIdx.y) * 8)) && ((((int)blockIdx.y) * 8) < ((33 - (((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 4)) + ax0_ax1_ax2_ax3_ax4_fused_fused_fused_fused_inner_inner_inner) % 128) / 16)) - rh_outer))) && (((1 - rw_outer) - (((((int)threadIdx.x) * 4) + ax0_ax1_ax2_ax3_ax4_fused_fused_fused_fused_inner_inner_inner) % 16)) <= (ww_inner_outer * 16))) && ((ww_inner_outer * 16) < ((33 - rw_outer) - (((((int)threadIdx.x) * 4) + ax0_ax1_ax2_ax3_ax4_fused_fused_fused_fused_inner_inner_inner) % 16)))) ? Data[(((((((((((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 4)) + ax0_ax1_ax2_ax3_ax4_fused_fused_fused_fused_inner_inner_inner) / 128) * 4096) + (rs1_outer * 1024)) + (((int)blockIdx.y) * 256)) + ((((((((int)threadIdx.y) * 64) + (((int)threadIdx.x) * 4)) + ax0_ax1_ax2_ax3_ax4_fused_fused_fused_fused_inner_inner_inner) % 128) / 16) * 32)) + (rh_outer * 32)) + (ww_inner_outer * 16)) + rw_outer) + (((((int)threadIdx.x) * 4) + ax0_ax1_ax2_ax3_ax4_fused_fused_fused_fused_inner_inner_inner) % 16)) - 33)] : 0.000000e+00f);
            }
            if (((int)threadIdx.x) < (8 - ((int)threadIdx.y))) {
              if (((int)threadIdx.x) < 1) {
                K0_shared[(((int)threadIdx.x) + ((int)threadIdx.y))] = K0[((((((((int)threadIdx.x) + ((int)threadIdx.y)) / 2) * 44) + ((((int)blockIdx.z) / 2) * 22)) + (((((int)threadIdx.x) + ((int)threadIdx.y)) % 2) * 11)) + rr_outer)];
              }
            }
            if (((int)threadIdx.x) < (2 - ((int)threadIdx.y))) {
              if (((int)threadIdx.x) < 1) {
                if (((((int)threadIdx.x) + ((int)threadIdx.y)) / 2) < (4 - rs1_outer)) {
                  K1_shared[(((int)threadIdx.x) + ((int)threadIdx.y))] = K1[(((((((((int)threadIdx.x) + ((int)threadIdx.y)) / 2) * 44) + (rs1_outer * 44)) + ((((int)blockIdx.z) % 2) * 22)) + (((((int)threadIdx.x) + ((int)threadIdx.y)) % 2) * 11)) + rr_outer)];
                }
              }
            }
            if (((int)threadIdx.x) < (1 - ((int)threadIdx.y))) {
              if (((int)threadIdx.x) < 1) {
                if (((int)threadIdx.x) < ((3 - rh_outer) - ((int)threadIdx.y))) {
                  KC_shared[(((int)threadIdx.x) + ((int)threadIdx.y))] = KC[(((((((int)threadIdx.x) * 33) + (((int)threadIdx.y) * 33)) + (rh_outer * 33)) + (rw_outer * 11)) + rr_outer)];
                }
              }
            }
            __syncthreads();
            #pragma unroll
            for (int rs0_inner = 0; rs0_inner < 4; ++rs0_inner) {
              #pragma unroll
              for (int tt0_c = 0; tt0_c < 2; ++tt0_c) {
                #pragma unroll
                for (int tt1_c = 0; tt1_c < 2; ++tt1_c) {
                  Output_local[((tt0_c * 2) + tt1_c)] = (Output_local[((tt0_c * 2) + tt1_c)] + (((pad_temp_shared[(((rs0_inner * 128) + (((int)threadIdx.y) * 16)) + ((int)threadIdx.x))] * K0_shared[((rs0_inner * 2) + tt0_c)]) * K1_shared[tt1_c]) * KC_shared[0]));
                }
              }
            }
          }
        }
      }
    }
    #pragma unroll
    for (int tt0_inner_inner_inner = 0; tt0_inner_inner_inner < 2; ++tt0_inner_inner_inner) {
      #pragma unroll
      for (int tt1_inner_inner_inner = 0; tt1_inner_inner_inner < 2; ++tt1_inner_inner_inner) {
        Output[(((((((((((int)blockIdx.z) / 2) * 8192) + (tt0_inner_inner_inner * 4096)) + ((((int)blockIdx.z) % 2) * 2048)) + (tt1_inner_inner_inner * 1024)) + (((int)blockIdx.y) * 256)) + (((int)threadIdx.y) * 32)) + (ww_inner_outer * 16)) + ((int)threadIdx.x))] = Output_local[((tt0_inner_inner_inner * 2) + tt1_inner_inner_inner)];
      }
    }
  }
}


void Conv2dRcpFusedNchwKernelLauncher(const float* U, const float* K0,
    const float* K1, const float* KC, float* V){

  dim3 gridDim0(1, 4, 4);
  dim3 blockDim0(16, 8, 1);

  default_function_kernel0<<<gridDim0, blockDim0>>>(U, K0, K1, KC, V);
  cudaDeviceSynchronize();

}

#endif
