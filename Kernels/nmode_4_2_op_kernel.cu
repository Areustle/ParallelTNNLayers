/* Best config: */
/* [('tile_r', [1, 1, 32, 2]), ('tile_y', [8, 1, 2, 4]), ('tile_x', [16, 1, 1, 4]), ('tile_rs', [32, 2, 1]), ('auto_unroll_max_step', 0), ('unroll_explicit', 1)],,None,50730168 */
/* Finish loading 23776 records */
/* Time cost of this operator: 0.000020 */

extern "C"
__global__
void default_function_kernel0( float* __restrict__ U,  float* __restrict__ K,  float* __restrict__ C) {
   float C_local[32];
  __shared__ float U_shared[64];
  __shared__ float K_shared[128];
   float U_shared_local[16];
   float K_shared_local[2];
  for (int x_c_init = 0; x_c_init < 4; ++x_c_init) {
    for (int y_c_init = 0; y_c_init < 4; ++y_c_init) {
      for (int r_c_init = 0; r_c_init < 2; ++r_c_init) {
        C_local[((x_c_init * 8) + (y_c_init * 2)) + r_c_init] = 0.0f;
      }
    }
  }
  for (int sdim_outer = 0; sdim_outer < 32; ++sdim_outer) {
    __syncthreads();

    int ty = (int)threadIdx.y;
    int tz_2 = (int)threadIdx.z * 2;
    int tz_strd = ((int)threadIdx.z / 8) * 4096;
    int tz_strd_2 = ((int)threadIdx.z % 8) * 64;
    int bx_16384 = (((int)blockIdx.x) * 16384);
    int by_512 = (int)blockIdx.y * 512;

    U_shared[tz_2 + ty] = U[(bx_16384 + tz_strd + by_512 + tz_strd_2) + ty + (sdim_outer * 2)];

    for (int ax0_ax1_fused_inner_inner_inner = 0; ax0_ax1_fused_inner_inner_inner < 2; ++ax0_ax1_fused_inner_inner_inner) {
      K_shared[(((((int)threadIdx.z) * 4) + (((int)threadIdx.y) * 2)) + ax0_ax1_fused_inner_inner_inner)] = K[((((sdim_outer * 128) + (((int)threadIdx.z) * 4)) + (((int)threadIdx.y) * 2)) + ax0_ax1_fused_inner_inner_inner)];
    }
    __syncthreads();
    for (int sdim_inner_outer = 0; sdim_inner_outer < 2; ++sdim_inner_outer) {
      for (int ax1 = 0; ax1 < 4; ++ax1) {
        for (int ax2 = 0; ax2 < 4; ++ax2) {
          U_shared_local[((ax1 * 4) + ax2)] = U_shared[((((ax1 * 16) + (((int)threadIdx.y) * 8)) + (ax2 * 2)) + sdim_inner_outer)];
        }
      }
      for (int ax11 = 0; ax11 < 2; ++ax11) {
        K_shared_local[ax11] = K_shared[(((sdim_inner_outer * 64) + (((int)threadIdx.z) * 2)) + ax11)];
      }
      for (int x_c = 0; x_c < 4; ++x_c) {
        for (int y_c = 0; y_c < 4; ++y_c) {
          for (int r_c = 0; r_c < 2; ++r_c) {
            C_local[(((x_c * 8) + (y_c * 2)) + r_c)] = (C_local[(((x_c * 8) + (y_c * 2)) + r_c)] + (U_shared_local[((x_c * 4) + y_c)] * K_shared_local[r_c]));
          }
        }
      }
    }
  }
  for (int x_inner_inner_inner = 0; x_inner_inner_inner < 4; ++x_inner_inner_inner) {
    for (int y_inner_inner_inner = 0; y_inner_inner_inner < 4; ++y_inner_inner_inner) {
      for (int r_inner_inner_inner = 0; r_inner_inner_inner < 2; ++r_inner_inner_inner) {
        C[(((((((((int)blockIdx.x) * 16384) + (x_inner_inner_inner * 4096)) + (((int)blockIdx.y) * 512)) + (((int)threadIdx.y) * 256)) + (y_inner_inner_inner * 64)) + (((int)threadIdx.z) * 2)) + r_inner_inner_inner)] = C_local[(((x_inner_inner_inner * 8) + (y_inner_inner_inner * 2)) + r_inner_inner_inner)];
      }
    }
  }
}
