#if GOOGLE_CUDA
#define EIGEN_USE_GPU

extern "C" __global__ void default_function_kernel0(const float* __restrict__ Data,
    const float* __restrict__ K0,
    const float* __restrict__ K1,
    const float* __restrict__ K2,
    float* __restrict__ Output) {

   float Output_local[16];
  __shared__ float pad_temp_shared[4096];
  __shared__ float K0_shared[16];
  __shared__ float K1_shared[1];
  __shared__ float K2_shared[2];
  for (int nn_inner_outer = 0; nn_inner_outer < 8; ++nn_inner_outer) {
    for (int hh_inner_outer = 0; hh_inner_outer < 8; ++hh_inner_outer) {
      for (int ww_inner_outer = 0; ww_inner_outer < 2; ++ww_inner_outer) {
        for (int nn_c_init = 0; nn_c_init < 4; ++nn_c_init) {
          for (int hh_c_init = 0; hh_c_init < 2; ++hh_c_init) {
            for (int ww_c_init = 0; ww_c_init < 2; ++ww_c_init) {
              Output_local[(((nn_c_init * 4) + (hh_c_init * 2)) + ww_c_init)] = 0.000000e+00f;
            }
          }
        }
        for (int rr_outer = 0; rr_outer < 6; ++rr_outer) {
          for (int rh_outer = 0; rh_outer < 3; ++rh_outer) {
            for (int rw_outer = 0; rw_outer < 3; ++rw_outer) {
              __syncthreads();
              for (int ax0_ax1_ax2_ax3_fused_fused_fused_inner_inner_inner = 0; ax0_ax1_ax2_ax3_fused_fused_fused_inner_inner_inner < 128; ++ax0_ax1_ax2_ax3_fused_fused_fused_inner_inner_inner) {
                pad_temp_shared[((((((int)threadIdx.z) * 2048) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.x) * 128)) + ax0_ax1_ax2_ax3_fused_fused_fused_inner_inner_inner)] = (((((((1 - ((ax0_ax1_ax2_ax3_fused_fused_fused_inner_inner_inner % 64) / 16)) - rh_outer) <= (hh_inner_outer * 4)) && ((hh_inner_outer * 4) < ((33 - ((ax0_ax1_ax2_ax3_fused_fused_fused_inner_inner_inner % 64) / 16)) - rh_outer))) && (((1 - rw_outer) - (ax0_ax1_ax2_ax3_fused_fused_fused_inner_inner_inner % 16)) <= (ww_inner_outer * 16))) && ((ww_inner_outer * 16) < ((33 - rw_outer) - (ax0_ax1_ax2_ax3_fused_fused_fused_inner_inner_inner % 16)))) ? Data[((((((((((((((((int)blockIdx.z) / 8) * 524288) + (nn_inner_outer * 65536)) + (((int)threadIdx.z) * 32768)) + (((int)threadIdx.y) * 16384)) + (((int)threadIdx.x) * 2048)) + ((ax0_ax1_ax2_ax3_fused_fused_fused_inner_inner_inner / 64) * 1024)) + (hh_inner_outer * 128)) + (((ax0_ax1_ax2_ax3_fused_fused_fused_inner_inner_inner % 64) / 16) * 32)) + (rh_outer * 32)) + (ww_inner_outer * 16)) + rw_outer) + (ax0_ax1_ax2_ax3_fused_fused_fused_inner_inner_inner % 16)) - 33)] : 0.000000e+00f);
              }
              if (((((int)threadIdx.z) * 8) + (((int)threadIdx.y) * 4)) < (16 - ((int)threadIdx.x))) {
                if ((((int)threadIdx.y) * 4) < (8 - ((int)threadIdx.x))) {
                  if (((int)threadIdx.x) < 4) {
                    K0_shared[(((((int)threadIdx.z) * 8) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x))] = K0[((((((int)threadIdx.z) * 48) + (((int)threadIdx.y) * 24)) + (((int)threadIdx.x) * 6)) + rr_outer)];
                  }
                }
              }
              if (((int)threadIdx.x) < ((1 - ((int)threadIdx.z)) - ((int)threadIdx.y))) {
                if (((int)threadIdx.x) < (1 - ((int)threadIdx.y))) {
                  if (((int)threadIdx.x) < 1) {
                    if (((int)threadIdx.x) < (((3 - rh_outer) - ((int)threadIdx.z)) - ((int)threadIdx.y))) {
                      K1_shared[((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z))] = K1[((((((((int)threadIdx.x) * 18) + (((int)threadIdx.y) * 18)) + (((int)threadIdx.z) * 18)) + (rh_outer * 18)) + (rw_outer * 6)) + rr_outer)];
                    }
                  }
                }
              }
              if (((int)threadIdx.x) < ((2 - ((int)threadIdx.z)) - ((int)threadIdx.y))) {
                if (((int)threadIdx.x) < (1 - ((int)threadIdx.y))) {
                  if (((int)threadIdx.x) < 1) {
                    if ((((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) / 2) < (6 - rr_outer)) {
                      K2_shared[((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z))] = K2[(((((((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) / 2) * 16) + (rr_outer * 16)) + ((((int)blockIdx.z) % 8) * 2)) + (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) % 2))];
                    }
                  }
                }
              }
              __syncthreads();
              for (int rc_inner = 0; rc_inner < 16; ++rc_inner) {
                for (int nn_c = 0; nn_c < 4; ++nn_c) {
                  for (int hh_c = 0; hh_c < 2; ++hh_c) {
                    for (int ww_c = 0; ww_c < 2; ++ww_c) {
                      Output_local[(((nn_c * 4) + (hh_c * 2)) + ww_c)] = (Output_local[(((nn_c * 4) + (hh_c * 2)) + ww_c)] + (((pad_temp_shared[((((((nn_c * 1024) + (rc_inner * 64)) + (((int)threadIdx.y) * 32)) + (hh_c * 16)) + (((int)threadIdx.x) * 2)) + ww_c)] * K0_shared[rc_inner]) * K1_shared[0]) * K2_shared[((int)threadIdx.z)]));
                    }
                  }
                }
              }
            }
          }
        }
        for (int nn_inner_inner_inner = 0; nn_inner_inner_inner < 4; ++nn_inner_inner_inner) {
          for (int hh_inner_inner_inner = 0; hh_inner_inner_inner < 2; ++hh_inner_inner_inner) {
            for (int ww_inner_inner_inner = 0; ww_inner_inner_inner < 2; ++ww_inner_inner_inner) {
              Output[((((((((((((((int)blockIdx.z) / 8) * 524288) + (nn_inner_outer * 65536)) + (nn_inner_inner_inner * 16384)) + ((((int)blockIdx.z) % 8) * 2048)) + (((int)threadIdx.z) * 1024)) + (hh_inner_outer * 128)) + (((int)threadIdx.y) * 64)) + (hh_inner_inner_inner * 32)) + (ww_inner_outer * 16)) + (((int)threadIdx.x) * 2)) + ww_inner_inner_inner)] = Output_local[(((nn_inner_inner_inner * 4) + (hh_inner_inner_inner * 2)) + ww_inner_inner_inner)];
            }
          }
        }
      }
    }
  }
}


void Conv2dCpFusedNchwKernelLauncher(const float* U, const float* K0,
    const float* K1, const float* K2, float* V){

  dim3 gridDim0(1, 1, 16);
  dim3 blockDim0(8, 2, 2);

  default_function_kernel0<<<gridDim0, blockDim0>>>(U, K0, K1, K2, V);
  cudaDeviceSynchronize();

}

#endif
