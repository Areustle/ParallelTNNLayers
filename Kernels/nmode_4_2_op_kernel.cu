/*
Best config:
[('tile_r', [1, 1, 64, 1]), ('tile_y', [8, 4, 1, 2]), ('tile_x', [16, 2, 1, 2]), ('tile_rs', [8, 1, 8]), ('auto_unroll_max_step', 512), ('unroll_explicit', 0)],,None,27471555
Finish loading 21712 records
Time cost of this operator: 0.000030
extern "C" __global__ void default_function_kernel0( float* __restrict__ U,  float* __restrict__ K,  float* __restrict__ C) {
   float C_local[32];
  __shared__ float U_shared[256];
  __shared__ float K_shared[512];
   float U_shared_local[256];
   float K_shared_local[8];
  #pragma unroll
  for (int x_c_init = 0; x_c_init < 2; ++x_c_init) {
    #pragma unroll
    for (int y_c_init = 0; y_c_init < 2; ++y_c_init) {
      C_local[((x_c_init * 2) + y_c_init)] = 0.000000e+00f;
      C_local[(((x_c_init * 2) + y_c_init) + 16)] = 0.000000e+00f;
      C_local[(((x_c_init * 2) + y_c_init) + 4)] = 0.000000e+00f;
      C_local[(((x_c_init * 2) + y_c_init) + 20)] = 0.000000e+00f;
      C_local[(((x_c_init * 2) + y_c_init) + 8)] = 0.000000e+00f;
      C_local[(((x_c_init * 2) + y_c_init) + 24)] = 0.000000e+00f;
      C_local[(((x_c_init * 2) + y_c_init) + 12)] = 0.000000e+00f;
      C_local[(((x_c_init * 2) + y_c_init) + 28)] = 0.000000e+00f;
    }
  }
  for (int sdim_outer = 0; sdim_outer < 8; ++sdim_outer) {
    __syncthreads();
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner < 4; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) {
      U_shared[((((((((int)threadIdx.z) * 4) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) / 256) * 256) + (((int)threadIdx.z) * 4)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner)] = U[((((((((((((int)threadIdx.z) * 4) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) / 256) * 262144) + (((int)blockIdx.x) * 16384)) + ((((((int)threadIdx.z) * 4) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) / 64) * 4096)) + (((int)blockIdx.y) * 512)) + (((((((int)threadIdx.z) * 4) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) % 64) / 8) * 64)) + (sdim_outer * 8)) + (((((int)threadIdx.z) * 4) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) % 8))];
    }
    #pragma unroll
    for (int ax0_ax1_fused_inner_inner_inner = 0; ax0_ax1_fused_inner_inner_inner < 8; ++ax0_ax1_fused_inner_inner_inner) {
      K_shared[((((int)threadIdx.z) * 8) + ax0_ax1_fused_inner_inner_inner)] = K[(((sdim_outer * 512) + (((int)threadIdx.z) * 8)) + ax0_ax1_fused_inner_inner_inner)];
    }
    __syncthreads();
    #pragma unroll
    for (int ax1 = 0; ax1 < 2; ++ax1) {
      #pragma unroll
      for (int ax2 = 0; ax2 < 2; ++ax2) {
        #pragma unroll
        for (int ax3 = 0; ax3 < 8; ++ax3) {
          U_shared_local[(((ax1 * 16) + (ax2 * 8)) + ax3)] = U_shared[(((ax1 * 64) + (ax2 * 8)) + ax3)];
          U_shared_local[((((ax1 * 16) + (ax2 * 8)) + ax3) + 128)] = U_shared[((((ax1 * 64) + (ax2 * 8)) + ax3) + 128)];
          U_shared_local[((((ax1 * 16) + (ax2 * 8)) + ax3) + 32)] = U_shared[((((ax1 * 64) + (ax2 * 8)) + ax3) + 16)];
          U_shared_local[((((ax1 * 16) + (ax2 * 8)) + ax3) + 160)] = U_shared[((((ax1 * 64) + (ax2 * 8)) + ax3) + 144)];
          U_shared_local[((((ax1 * 16) + (ax2 * 8)) + ax3) + 64)] = U_shared[((((ax1 * 64) + (ax2 * 8)) + ax3) + 32)];
          U_shared_local[((((ax1 * 16) + (ax2 * 8)) + ax3) + 192)] = U_shared[((((ax1 * 64) + (ax2 * 8)) + ax3) + 160)];
          U_shared_local[((((ax1 * 16) + (ax2 * 8)) + ax3) + 96)] = U_shared[((((ax1 * 64) + (ax2 * 8)) + ax3) + 48)];
          U_shared_local[((((ax1 * 16) + (ax2 * 8)) + ax3) + 224)] = U_shared[((((ax1 * 64) + (ax2 * 8)) + ax3) + 176)];
        }
      }
    }
    #pragma unroll
    for (int ax0 = 0; ax0 < 8; ++ax0) {
      K_shared_local[ax0] = K_shared[((ax0 * 64) + ((int)threadIdx.z))];
    }
    #pragma unroll
    for (int sdim_inner_inner = 0; sdim_inner_inner < 8; ++sdim_inner_inner) {
      #pragma unroll
      for (int x_c = 0; x_c < 2; ++x_c) {
        #pragma unroll
        for (int y_c = 0; y_c < 2; ++y_c) {
          C_local[((x_c * 2) + y_c)] = (C_local[((x_c * 2) + y_c)] + (U_shared_local[(((x_c * 16) + (y_c * 8)) + sdim_inner_inner)] * K_shared_local[sdim_inner_inner]));
          C_local[(((x_c * 2) + y_c) + 16)] = (C_local[(((x_c * 2) + y_c) + 16)] + (U_shared_local[((((x_c * 16) + (y_c * 8)) + sdim_inner_inner) + 128)] * K_shared_local[sdim_inner_inner]));
          C_local[(((x_c * 2) + y_c) + 4)] = (C_local[(((x_c * 2) + y_c) + 4)] + (U_shared_local[((((x_c * 16) + (y_c * 8)) + sdim_inner_inner) + 32)] * K_shared_local[sdim_inner_inner]));
          C_local[(((x_c * 2) + y_c) + 20)] = (C_local[(((x_c * 2) + y_c) + 20)] + (U_shared_local[((((x_c * 16) + (y_c * 8)) + sdim_inner_inner) + 160)] * K_shared_local[sdim_inner_inner]));
          C_local[(((x_c * 2) + y_c) + 8)] = (C_local[(((x_c * 2) + y_c) + 8)] + (U_shared_local[((((x_c * 16) + (y_c * 8)) + sdim_inner_inner) + 64)] * K_shared_local[sdim_inner_inner]));
          C_local[(((x_c * 2) + y_c) + 24)] = (C_local[(((x_c * 2) + y_c) + 24)] + (U_shared_local[((((x_c * 16) + (y_c * 8)) + sdim_inner_inner) + 192)] * K_shared_local[sdim_inner_inner]));
          C_local[(((x_c * 2) + y_c) + 12)] = (C_local[(((x_c * 2) + y_c) + 12)] + (U_shared_local[((((x_c * 16) + (y_c * 8)) + sdim_inner_inner) + 96)] * K_shared_local[sdim_inner_inner]));
          C_local[(((x_c * 2) + y_c) + 28)] = (C_local[(((x_c * 2) + y_c) + 28)] + (U_shared_local[((((x_c * 16) + (y_c * 8)) + sdim_inner_inner) + 224)] * K_shared_local[sdim_inner_inner]));
        }
      }
    }
  }
  #pragma unroll
  for (int x_inner_inner_inner = 0; x_inner_inner_inner < 2; ++x_inner_inner_inner) {
    #pragma unroll
    for (int y_inner_inner_inner = 0; y_inner_inner_inner < 2; ++y_inner_inner_inner) {
      C[(((((((int)blockIdx.x) * 16384) + (x_inner_inner_inner * 4096)) + (((int)blockIdx.y) * 512)) + (y_inner_inner_inner * 64)) + ((int)threadIdx.z))] = C_local[((x_inner_inner_inner * 2) + y_inner_inner_inner)];
      C[((((((((int)blockIdx.x) * 16384) + (x_inner_inner_inner * 4096)) + (((int)blockIdx.y) * 512)) + (y_inner_inner_inner * 64)) + ((int)threadIdx.z)) + 8192)] = C_local[(((x_inner_inner_inner * 2) + y_inner_inner_inner) + 16)];
      C[((((((((int)blockIdx.x) * 16384) + (x_inner_inner_inner * 4096)) + (((int)blockIdx.y) * 512)) + (y_inner_inner_inner * 64)) + ((int)threadIdx.z)) + 128)] = C_local[(((x_inner_inner_inner * 2) + y_inner_inner_inner) + 4)];
      C[((((((((int)blockIdx.x) * 16384) + (x_inner_inner_inner * 4096)) + (((int)blockIdx.y) * 512)) + (y_inner_inner_inner * 64)) + ((int)threadIdx.z)) + 8320)] = C_local[(((x_inner_inner_inner * 2) + y_inner_inner_inner) + 20)];
      C[((((((((int)blockIdx.x) * 16384) + (x_inner_inner_inner * 4096)) + (((int)blockIdx.y) * 512)) + (y_inner_inner_inner * 64)) + ((int)threadIdx.z)) + 256)] = C_local[(((x_inner_inner_inner * 2) + y_inner_inner_inner) + 8)];
      C[((((((((int)blockIdx.x) * 16384) + (x_inner_inner_inner * 4096)) + (((int)blockIdx.y) * 512)) + (y_inner_inner_inner * 64)) + ((int)threadIdx.z)) + 8448)] = C_local[(((x_inner_inner_inner * 2) + y_inner_inner_inner) + 24)];
      C[((((((((int)blockIdx.x) * 16384) + (x_inner_inner_inner * 4096)) + (((int)blockIdx.y) * 512)) + (y_inner_inner_inner * 64)) + ((int)threadIdx.z)) + 384)] = C_local[(((x_inner_inner_inner * 2) + y_inner_inner_inner) + 12)];
      C[((((((((int)blockIdx.x) * 16384) + (x_inner_inner_inner * 4096)) + (((int)blockIdx.y) * 512)) + (y_inner_inner_inner * 64)) + ((int)threadIdx.z)) + 8576)] = C_local[(((x_inner_inner_inner * 2) + y_inner_inner_inner) + 28)];
    }
  }
}
*/
