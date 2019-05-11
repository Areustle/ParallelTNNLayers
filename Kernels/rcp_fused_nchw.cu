#if GOOGLE_CUDA
#define EIGEN_USE_GPU

__global__ void default_function_kernel0(const float* __restrict__ Data,
    const float* __restrict__ K0,
    const float* __restrict__ K1,
    const float* __restrict__ KC,
    float* __restrict__ Output) {

     float Output_local[4];
  __shared__ float pad_temp_shared[1024];
  __shared__ float K0_shared[8];
  __shared__ float K1_shared[8];
  __shared__ float KC_shared[1];
  for (int oca_inner_outer = 0; oca_inner_outer < 2; ++oca_inner_outer) {
    for (int hh_inner_outer = 0; hh_inner_outer < 2; ++hh_inner_outer) {
      #pragma unroll
      for (int oca_c_init = 0; oca_c_init < 2; ++oca_c_init) {
        #pragma unroll
        for (int ocb_c_init = 0; ocb_c_init < 2; ++ocb_c_init) {
          Output_local[((oca_c_init * 2) + ocb_c_init)] = 0.000000e+00f;
        }
      }
      for (int rr_outer = 0; rr_outer < 11; ++rr_outer) {
        for (int rh_outer = 0; rh_outer < 3; ++rh_outer) {
          #pragma unroll
          for (int rw_outer = 0; rw_outer < 3; ++rw_outer) {
            __syncthreads();
            #pragma unroll
            for (int ax0_ax1_ax2_ax3_ax4_fused_fused_fused_fused_inner_inner_inner = 0; ax0_ax1_ax2_ax3_ax4_fused_fused_fused_fused_inner_inner_inner < 16; ++ax0_ax1_ax2_ax3_ax4_fused_fused_fused_fused_inner_inner_inner) {
              pad_temp_shared[(((((int)threadIdx.y) * 512) + (((int)threadIdx.x) * 16)) + ax0_ax1_ax2_ax3_ax4_fused_fused_fused_fused_inner_inner_inner)] = (((((((1 - ((((((int)threadIdx.x) * 16) + ax0_ax1_ax2_ax3_ax4_fused_fused_fused_fused_inner_inner_inner) % 64) / 32)) - rh_outer) <= ((((int)blockIdx.y) * 4) + (hh_inner_outer * 2))) && (((((int)blockIdx.y) * 4) + (hh_inner_outer * 2)) < ((33 - ((((((int)threadIdx.x) * 16) + ax0_ax1_ax2_ax3_ax4_fused_fused_fused_fused_inner_inner_inner) % 64) / 32)) - rh_outer))) && ((1 - (((((int)threadIdx.x) * 16) + ax0_ax1_ax2_ax3_ax4_fused_fused_fused_fused_inner_inner_inner) % 32)) <= rw_outer)) && (rw_outer < (33 - (((((int)threadIdx.x) * 16) + ax0_ax1_ax2_ax3_ax4_fused_fused_fused_fused_inner_inner_inner) % 32)))) ? Data[((((((((((int)threadIdx.y) * 8192) + ((((((int)threadIdx.x) * 16) + ax0_ax1_ax2_ax3_ax4_fused_fused_fused_fused_inner_inner_inner) / 64) * 1024)) + (((int)blockIdx.y) * 128)) + (hh_inner_outer * 64)) + (rh_outer * 32)) + rw_outer) + (((((int)threadIdx.x) * 16) + ax0_ax1_ax2_ax3_ax4_fused_fused_fused_fused_inner_inner_inner) % 64)) - 33)] : 0.000000e+00f);
            }
            if ((((int)threadIdx.y) * 4) < (8 - ((int)threadIdx.x))) {
              if (((int)threadIdx.x) < 4) {
                if ((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) / 8) < (4 - rr_outer)) {
                  K0_shared[((((int)threadIdx.y) * 4) + ((int)threadIdx.x))] = K0[(((((rr_outer * 44) + (((int)threadIdx.y) * 22)) + ((((int)threadIdx.x) / 2) * 11)) + (oca_inner_outer * 2)) + (((int)threadIdx.x) % 2))];
                }
              }
            }
            if ((((int)threadIdx.y) * 4) < (8 - ((int)threadIdx.x))) {
              if (((int)threadIdx.x) < 4) {
                if ((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) / 8) < (4 - rr_outer)) {
                  K1_shared[((((int)threadIdx.y) * 4) + ((int)threadIdx.x))] = K1[(((((rr_outer * 44) + (((int)threadIdx.y) * 22)) + ((((int)threadIdx.x) / 2) * 11)) + (((int)blockIdx.z) * 2)) + (((int)threadIdx.x) % 2))];
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
            for (int rca_inner = 0; rca_inner < 4; ++rca_inner) {
              #pragma unroll
              for (int rcb_inner = 0; rcb_inner < 4; ++rcb_inner) {
                #pragma unroll
                for (int oca_c = 0; oca_c < 2; ++oca_c) {
                  #pragma unroll
                  for (int ocb_c = 0; ocb_c < 2; ++ocb_c) {
                    Output_local[((oca_c * 2) + ocb_c)] = (Output_local[((oca_c * 2) + ocb_c)] + (((pad_temp_shared[((((rca_inner * 256) + (rcb_inner * 64)) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[((rca_inner * 2) + oca_c)]) * K1_shared[((rcb_inner * 2) + ocb_c)]) * KC_shared[0]));
                  }
                }
              }
            }
          }
        }
      }
      #pragma unroll
      for (int oca_inner_inner_inner = 0; oca_inner_inner_inner < 2; ++oca_inner_inner_inner) {
        #pragma unroll
        for (int ocb_inner_inner_inner = 0; ocb_inner_inner_inner < 2; ++ocb_inner_inner_inner) {
          Output[((((((((oca_inner_outer * 8192) + (oca_inner_inner_inner * 4096)) + (((int)blockIdx.z) * 2048)) + (ocb_inner_inner_inner * 1024)) + (((int)blockIdx.y) * 128)) + (hh_inner_outer * 64)) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] = Output_local[((oca_inner_inner_inner * 2) + ocb_inner_inner_inner)];
        }
      }
    }
  }
}

void Conv2dRcpFusedNchwKernelLauncher(const float* U, const float* K0,
    const float* K1, const float* KC, float* V){

  dim3 gridDim0(1, 8, 2);
  dim3 blockDim0(32, 2, 1);

  default_function_kernel0<<<gridDim0, blockDim0>>>(U, K0, K1, KC, V);
  cudaDeviceSynchronize();

}

#endif
