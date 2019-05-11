#if GOOGLE_CUDA
#define EIGEN_USE_GPU

__global__ void default_function_kernel0(const float* __restrict__ Data,
    const float* __restrict__ K0,
    const float* __restrict__ K1,
    const float* __restrict__ KC,
    float* __restrict__ Output) {

   float Output_local[128];
  __shared__ float pad_temp_shared[640];
  __shared__ float K0_shared[8];
  __shared__ float K1_shared[1];
  __shared__ float KC_shared[3];
  for (int ww_inner_outer = 0; ww_inner_outer < 2; ++ww_inner_outer) {
    for (int nn_c_init = 0; nn_c_init < 2; ++nn_c_init) {
      for (int oca_c_init = 0; oca_c_init < 4; ++oca_c_init) {
        for (int hh_c_init = 0; hh_c_init < 8; ++hh_c_init) {
          for (int ww_c_init = 0; ww_c_init < 2; ++ww_c_init) {
            Output_local[((((nn_c_init * 64) + (oca_c_init * 16)) + (hh_c_init * 2)) + ww_c_init)] = 0.000000e+00f;
          }
        }
      }
    }
    for (int rr_outer = 0; rr_outer < 11; ++rr_outer) {
      for (int rca_outer = 0; rca_outer < 2; ++rca_outer) {
        for (int rcb_outer = 0; rcb_outer < 4; ++rcb_outer) {
          for (int rw_outer = 0; rw_outer < 3; ++rw_outer) {
            __syncthreads();
            for (int ax0_ax1_ax2_ax3_ax4_fused_fused_fused_fused_inner_inner_inner = 0; ax0_ax1_ax2_ax3_ax4_fused_fused_fused_fused_inner_inner_inner < 80; ++ax0_ax1_ax2_ax3_ax4_fused_fused_fused_fused_inner_inner_inner) {
              pad_temp_shared[(((((int)threadIdx.z) * 320) + (((int)threadIdx.x) * 80)) + ax0_ax1_ax2_ax3_ax4_fused_fused_fused_fused_inner_inner_inner)] = ((((((1 - (ax0_ax1_ax2_ax3_ax4_fused_fused_fused_fused_inner_inner_inner / 8)) <= (((int)blockIdx.y) * 8)) && ((((int)blockIdx.y) * 8) < (33 - (ax0_ax1_ax2_ax3_ax4_fused_fused_fused_fused_inner_inner_inner / 8)))) && (((1 - rw_outer) - (ax0_ax1_ax2_ax3_ax4_fused_fused_fused_fused_inner_inner_inner % 8)) <= ((((int)blockIdx.x) * 16) + (ww_inner_outer * 8)))) && (((((int)blockIdx.x) * 16) + (ww_inner_outer * 8)) < ((33 - rw_outer) - (ax0_ax1_ax2_ax3_ax4_fused_fused_fused_fused_inner_inner_inner % 8)))) ? Data[((((((((((((((((int)blockIdx.z) / 4) * 65536) + (((int)threadIdx.z) * 32768)) + ((((((int)threadIdx.x) * 80) + ax0_ax1_ax2_ax3_ax4_fused_fused_fused_fused_inner_inner_inner) / 160) * 16384)) + (rca_outer * 8192)) + (((((((int)threadIdx.x) * 80) + ax0_ax1_ax2_ax3_ax4_fused_fused_fused_fused_inner_inner_inner) % 160) / 80) * 4096)) + (rcb_outer * 1024)) + (((int)blockIdx.y) * 256)) + ((ax0_ax1_ax2_ax3_ax4_fused_fused_fused_fused_inner_inner_inner / 8) * 32)) + (((int)blockIdx.x) * 16)) + (ww_inner_outer * 8)) + rw_outer) + (ax0_ax1_ax2_ax3_ax4_fused_fused_fused_fused_inner_inner_inner % 8)) - 33)] : 0.000000e+00f);
            }
            if ((((((int)threadIdx.z) * 4) + ((int)threadIdx.x)) / 8) < (4 - rr_outer)) {
              K0_shared[((((int)threadIdx.z) * 4) + ((int)threadIdx.x))] = K0[((((((((((int)threadIdx.z) * 4) + ((int)threadIdx.x)) / 8) * 44) + (rr_outer * 44)) + (rca_outer * 22)) + (((((((int)threadIdx.z) * 4) + ((int)threadIdx.x)) % 8) / 4) * 11)) + ((int)threadIdx.x))];
            }
            if (((int)threadIdx.x) < (1 - ((int)threadIdx.z))) {
              if (((int)threadIdx.x) < 1) {
                if (((int)threadIdx.x) < ((4 - rr_outer) - ((int)threadIdx.z))) {
                  K1_shared[(((int)threadIdx.x) + ((int)threadIdx.z))] = K1[(((((((int)threadIdx.x) * 44) + (((int)threadIdx.z) * 44)) + (rr_outer * 44)) + (rcb_outer * 11)) + (((int)blockIdx.z) % 4))];
                }
              }
            }
            if ((((int)threadIdx.z) * 2) < (3 - ((int)threadIdx.x))) {
              if (((int)threadIdx.x) < 2) {
                KC_shared[((((int)threadIdx.z) * 2) + ((int)threadIdx.x))] = KC[((((((int)threadIdx.z) * 66) + (((int)threadIdx.x) * 33)) + (rw_outer * 11)) + rr_outer)];
              }
            }
            __syncthreads();
            for (int rca_inner = 0; rca_inner < 2; ++rca_inner) {
              for (int rh_inner = 0; rh_inner < 3; ++rh_inner) {
                for (int nn_c = 0; nn_c < 2; ++nn_c) {
                  for (int oca_c = 0; oca_c < 4; ++oca_c) {
                    for (int hh_c = 0; hh_c < 8; ++hh_c) {
                      for (int ww_c = 0; ww_c < 2; ++ww_c) {
                        Output_local[((((nn_c * 64) + (oca_c * 16)) + (hh_c * 2)) + ww_c)] = (Output_local[((((nn_c * 64) + (oca_c * 16)) + (hh_c * 2)) + ww_c)] + (((pad_temp_shared[(((((((((int)threadIdx.z) * 320) + (nn_c * 160)) + (rca_inner * 80)) + (hh_c * 8)) + (rh_inner * 8)) + (((int)threadIdx.x) * 2)) + ww_c)] * K0_shared[((rca_inner * 4) + oca_c)]) * K1_shared[0]) * KC_shared[rh_inner]));
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    for (int nn_inner_inner_inner = 0; nn_inner_inner_inner < 2; ++nn_inner_inner_inner) {
      for (int oca_inner_inner_inner = 0; oca_inner_inner_inner < 4; ++oca_inner_inner_inner) {
        for (int hh_inner_inner_inner = 0; hh_inner_inner_inner < 8; ++hh_inner_inner_inner) {
          for (int ww_inner_inner_inner = 0; ww_inner_inner_inner < 2; ++ww_inner_inner_inner) {
            Output[((((((((((((((int)blockIdx.z) / 4) * 65536) + (((int)threadIdx.z) * 32768)) + (nn_inner_inner_inner * 16384)) + (oca_inner_inner_inner * 4096)) + ((((int)blockIdx.z) % 4) * 1024)) + (((int)blockIdx.y) * 256)) + (hh_inner_inner_inner * 32)) + (((int)blockIdx.x) * 16)) + (ww_inner_outer * 8)) + (((int)threadIdx.x) * 2)) + ww_inner_inner_inner)] = Output_local[((((nn_inner_inner_inner * 64) + (oca_inner_inner_inner * 16)) + (hh_inner_inner_inner * 2)) + ww_inner_inner_inner)];
          }
        }
      }
    }
  }
}


void Conv2dRcpFusedNchwKernelLauncher(const float* U, const float* K0,
    const float* K1, const float* KC, float* V){

  dim3 gridDim0(2, 4, 8);
  dim3 blockDim0(4, 1, 2);

  default_function_kernel0<<<gridDim0, blockDim0>>>(U, K0, K1, KC, V);
  cudaDeviceSynchronize();

}

#endif
