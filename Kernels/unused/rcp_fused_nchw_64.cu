#if GOOGLE_CUDA
#define EIGEN_USE_GPU

__global__ void default_function_kernel0(const float* __restrict__ Data,
    const float* __restrict__ K0,
    const float* __restrict__ K1,
    const float* __restrict__ KC,
    float* __restrict__ Output) {

  float Output_local[4];
  __shared__ float pad_temp_shared[64];
  __shared__ float K0_shared[4];
  __shared__ float K1_shared[1];
  __shared__ float KC_shared[3];
  for (int nn_inner_outer = 0; nn_inner_outer < 4; ++nn_inner_outer) {
    for (int ocb_inner_outer = 0; ocb_inner_outer < 2; ++ocb_inner_outer) {
      for (int oca_c_init = 0; oca_c_init < 2; ++oca_c_init) {
        for (int ww_c_init = 0; ww_c_init < 2; ++ww_c_init) {
          Output_local[((oca_c_init * 2) + ww_c_init)] = 0.000000e+00f;
        }
      }
      for (int rr_outer = 0; rr_outer < 11; ++rr_outer) {
        for (int rca_outer = 0; rca_outer < 2; ++rca_outer) {
          for (int rcb_outer = 0; rcb_outer < 4; ++rcb_outer) {
            for (int rw_outer = 0; rw_outer < 3; ++rw_outer) {
              __syncthreads();
              for (int ax0_ax1_ax2_ax3_ax4_fused_fused_fused_fused_inner_inner_inner = 0; ax0_ax1_ax2_ax3_ax4_fused_fused_fused_fused_inner_inner_inner < 8; ++ax0_ax1_ax2_ax3_ax4_fused_fused_fused_fused_inner_inner_inner) {
                pad_temp_shared[(((((int)threadIdx.y) * 32) + (((int)threadIdx.x) * 8)) + ax0_ax1_ax2_ax3_ax4_fused_fused_fused_fused_inner_inner_inner)] = ((((((1 - (((((int)threadIdx.x) * 8) + ax0_ax1_ax2_ax3_ax4_fused_fused_fused_fused_inner_inner_inner) / 8)) <= (((int)blockIdx.y) * 2)) && ((((int)blockIdx.y) * 2) < (33 - (((((int)threadIdx.x) * 8) + ax0_ax1_ax2_ax3_ax4_fused_fused_fused_fused_inner_inner_inner) / 8)))) && (((1 - ax0_ax1_ax2_ax3_ax4_fused_fused_fused_fused_inner_inner_inner) - rw_outer) <= (((int)blockIdx.x) * 8))) && ((((int)blockIdx.x) * 8) < ((33 - ax0_ax1_ax2_ax3_ax4_fused_fused_fused_fused_inner_inner_inner) - rw_outer))) ? Data[(((((((((((((((int)blockIdx.z) / 4) * 65536) + (((((((int)threadIdx.y) * 32) + (((int)threadIdx.x) * 8)) + ax0_ax1_ax2_ax3_ax4_fused_fused_fused_fused_inner_inner_inner) / 64) * 16384)) + (nn_inner_outer * 16384)) + (rca_outer * 8192)) + ((((((((int)threadIdx.y) * 32) + (((int)threadIdx.x) * 8)) + ax0_ax1_ax2_ax3_ax4_fused_fused_fused_fused_inner_inner_inner) % 64) / 32) * 4096)) + (rcb_outer * 1024)) + (((int)blockIdx.y) * 64)) + ((((((int)threadIdx.x) * 8) + ax0_ax1_ax2_ax3_ax4_fused_fused_fused_fused_inner_inner_inner) / 8) * 32)) + (((int)blockIdx.x) * 8)) + rw_outer) + ax0_ax1_ax2_ax3_ax4_fused_fused_fused_fused_inner_inner_inner) - 33)] : 0.000000e+00f);
              }
              if ((((int)threadIdx.y) * 2) < (4 - ((int)threadIdx.x))) {
                if (((int)threadIdx.x) < 2) {
                  if ((((((int)threadIdx.y) * 2) + ((int)threadIdx.x)) / 4) < (4 - rr_outer)) {
                    K0_shared[((((int)threadIdx.y) * 2) + ((int)threadIdx.x))] = K0[(((((((((((int)threadIdx.y) * 2) + ((int)threadIdx.x)) / 4) * 44) + (rr_outer * 44)) + (rca_outer * 22)) + (((((((int)threadIdx.y) * 2) + ((int)threadIdx.x)) % 4) / 2) * 11)) + (((((int)blockIdx.z) % 4) / 2) * 2)) + (((int)threadIdx.x) % 2))];
                  }
                }
              }
              if (((int)threadIdx.x) < (1 - ((int)threadIdx.y))) {
                if (((int)threadIdx.x) < 1) {
                  if (((int)threadIdx.x) < ((4 - rr_outer) - ((int)threadIdx.y))) {
                    K1_shared[(((int)threadIdx.x) + ((int)threadIdx.y))] = K1[((((((((int)threadIdx.x) * 44) + (((int)threadIdx.y) * 44)) + (rr_outer * 44)) + (rcb_outer * 11)) + ((((int)blockIdx.z) % 2) * 2)) + ocb_inner_outer)];
                  }
                }
              }
              if ((((int)threadIdx.y) * 2) < (3 - ((int)threadIdx.x))) {
                if (((int)threadIdx.x) < 2) {
                  KC_shared[((((int)threadIdx.y) * 2) + ((int)threadIdx.x))] = KC[((((((int)threadIdx.y) * 66) + (((int)threadIdx.x) * 33)) + (rw_outer * 11)) + rr_outer)];
                }
              }
              __syncthreads();
              for (int rca_inner = 0; rca_inner < 2; ++rca_inner) {
                for (int rh_inner = 0; rh_inner < 3; ++rh_inner) {
                  for (int oca_c = 0; oca_c < 2; ++oca_c) {
                    for (int ww_c = 0; ww_c < 2; ++ww_c) {
                      Output_local[((oca_c * 2) + ww_c)] = (Output_local[((oca_c * 2) + ww_c)] + (((pad_temp_shared[(((((rca_inner * 32) + (((int)threadIdx.y) * 8)) + (rh_inner * 8)) + (((int)threadIdx.x) * 2)) + ww_c)] * K0_shared[((rca_inner * 2) + oca_c)]) * K1_shared[0]) * KC_shared[rh_inner]));
                    }
                  }
                }
              }
            }
          }
        }
      }
      for (int oca_inner_inner_inner = 0; oca_inner_inner_inner < 2; ++oca_inner_inner_inner) {
        for (int ww_inner_inner_inner = 0; ww_inner_inner_inner < 2; ++ww_inner_inner_inner) {
          Output[((((((((((((((int)blockIdx.z) / 4) * 65536) + (nn_inner_outer * 16384)) + (((((int)blockIdx.z) % 4) / 2) * 8192)) + (oca_inner_inner_inner * 4096)) + ((((int)blockIdx.z) % 2) * 2048)) + (ocb_inner_outer * 1024)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 32)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 2)) + ww_inner_inner_inner)] = Output_local[((oca_inner_inner_inner * 2) + ww_inner_inner_inner)];
        }
      }
    }
  }
}

void Conv2dRcpFusedNchwKernelLauncher(const float* U, const float* K0,
    const float* K1, const float* KC, float* V){

  dim3 gridDim0(4, 16, 64);
  dim3 blockDim0(4, 2, 1);

  default_function_kernel0<<<gridDim0, blockDim0>>>(U, K0, K1, KC, V);
  cudaDeviceSynchronize();

}

#endif
