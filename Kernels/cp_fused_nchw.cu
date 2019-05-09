#if GOOGLE_CUDA
#define EIGEN_USE_GPU

extern "C" __global__ void default_function_kernel0(const float* __restrict__ Data,
    const float* __restrict__ K0,
    const float* __restrict__ K1,
    const float* __restrict__ K2,
    float* __restrict__ Output) {

   float Output_local[2];
  __shared__ float pad_temp_shared[1920];
  __shared__ float K0_shared[8];
  __shared__ float K1_shared[18];
  __shared__ float K2_shared[16];
  for (int hhio = 0; hhio < 2; ++hhio) {
    for (int ko_c_init = 0; ko_c_init < 2; ++ko_c_init) {
      Output_local[ko_c_init] = 0.000000e+00f;
    }
    for (int rr_outer = 0; rr_outer < 3; ++rr_outer) {
      for (int rc_outer = 0; rc_outer < 4; ++rc_outer) {
        __syncthreads();
        for (int axis = 0; axis < 2; ++axis) {
          if ((((int)threadIdx.z) * 60) < (((1920 - axis) - (((int)threadIdx.x) * 2)) - (((int)threadIdx.y) * 8))) {
            if (((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) < (60 - axis)) {
              pad_temp_shared[((((((int)threadIdx.z) * 60) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + axis)] = ((((((1 - (((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + axis) % 60) / 6)) <= ((((int)blockIdx.y) * 16) + (hhio * 8))) && (((((int)blockIdx.y) * 16) + (hhio * 8)) < (33 - (((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + axis) % 60) / 6)))) && ((1 - ((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + axis) % 6)) <= (((int)blockIdx.x) * 4))) && ((((int)blockIdx.x) * 4) < (33 - ((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + axis) % 6)))) ? Data[((((((((((((((((int)threadIdx.z) * 60) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + axis) / 240) * 16384) + (rc_outer * 4096)) + (((((((((int)threadIdx.z) * 60) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) * 2)) + axis) % 240) / 60) * 1024)) + (((int)blockIdx.y) * 512)) + (hhio * 256)) + ((((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + axis) % 60) / 6) * 32)) + (((int)blockIdx.x) * 4)) + ((((((int)threadIdx.y) * 8) + (((int)threadIdx.x) * 2)) + axis) % 6)) - 33)] : 0.000000e+00f);
            }
          }
        }
        if (((int)threadIdx.x) < ((8 - ((int)threadIdx.z)) - ((int)threadIdx.y))) {
          if (((int)threadIdx.x) < (1 - ((int)threadIdx.y))) {
            if (((int)threadIdx.x) < 1) {
              if ((rc_outer * 4) < (16 - (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) / 2))) {
                K0_shared[((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z))] = K0[((((rc_outer * 24) + ((((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) / 2) * 6)) + (rr_outer * 2)) + (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) % 2))];
              }
            }
          }
        }
        if (((int)threadIdx.x) < ((18 - ((int)threadIdx.z)) - ((int)threadIdx.y))) {
          if (((int)threadIdx.x) < (1 - ((int)threadIdx.y))) {
            if (((int)threadIdx.x) < 1) {
              K1_shared[((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z))] = K1[((((((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) / 2) * 6) + (rr_outer * 2)) + (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) % 2))];
            }
          }
        }
        if (((int)threadIdx.x) < ((16 - ((int)threadIdx.z)) - ((int)threadIdx.y))) {
          if (((int)threadIdx.x) < (1 - ((int)threadIdx.y))) {
            if (((int)threadIdx.x) < 1) {
              if ((rr_outer * 2) < (6 - (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) / 8))) {
                K2_shared[((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z))] = K2[((((rr_outer * 32) + ((((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) / 8) * 16)) + (((int)blockIdx.z) * 8)) + (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) % 8))];
              }
            }
          }
        }
        __syncthreads();
        for (int rr_inner = 0; rr_inner < 2; ++rr_inner) {
          for (int rc_inner = 0; rc_inner < 4; ++rc_inner) {
            for (int rh_inner = 0; rh_inner < 3; ++rh_inner) {
              for (int rw_inner = 0; rw_inner < 3; ++rw_inner) {
                for (int ko_c = 0; ko_c < 2; ++ko_c) {
                  Output_local[ko_c] = (Output_local[ko_c] + (((pad_temp_shared[(((((((((int)threadIdx.z) / 4) * 240) + (rc_inner * 60)) + (((int)threadIdx.y) * 6)) + (rh_inner * 6)) + ((int)threadIdx.x)) + rw_inner)] * K0_shared[((rc_inner * 2) + rr_inner)]) * K1_shared[(((rh_inner * 6) + (rw_inner * 2)) + rr_inner)]) * K2_shared[(((rr_inner * 8) + ((((int)threadIdx.z) % 4) * 2)) + ko_c)]));
                }
              }
            }
          }
        }
      }
    }
    for (int ko_inner_inner_inner = 0; ko_inner_inner_inner < 2; ++ko_inner_inner_inner) {
      Output[((((((((((((int)threadIdx.z) / 4) * 16384) + (((int)blockIdx.z) * 8192)) + ((((int)threadIdx.z) % 4) * 2048)) + (ko_inner_inner_inner * 1024)) + (((int)blockIdx.y) * 512)) + (hhio * 256)) + (((int)threadIdx.y) * 32)) + (((int)blockIdx.x) * 4)) + ((int)threadIdx.x))] = Output_local[ko_inner_inner_inner];
    }
  }
}

void Conv2dCpFusedNchwKernelLauncher(const float* U, const float* K0,
    const float* K1, const float* K2, float* V){

  dim3 gridDim0(8, 2, 2);
  dim3 blockDim0(4, 8, 32);

  default_function_kernel0<<<gridDim0, blockDim0>>>(U, K0, K1, K2, V);
  cudaDeviceSynchronize();

}

#endif
