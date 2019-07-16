#include <iostream>


__global__
void default_function_kernel0(const float* __restrict__ Data,
    const float* __restrict__ K0,
    const float* __restrict__ K1,
    const float* __restrict__ K2,
    const float* __restrict__ K3,
    float* __restrict__ Output) {

   float Output_local[2];
  __shared__ float pad_temp_shared[272];
  __shared__ float K0_shared[4];
  __shared__ float K1_shared[1];
  __shared__ float K2_shared[3];
  __shared__ float K3_shared[16];
   float pad_temp_shared_local[6];
   float K0_shared_local[1];
   float K1_shared_local[1];
   float K2_shared_local[3];
   float K3_shared_local[1];
  for (int hh_c_init = 0; hh_c_init < 2; ++hh_c_init) {
    Output_local[hh_c_init] = 0.000000e+00f;
  }
  for (int rr_outer = 0; rr_outer < 6; ++rr_outer) {
    for (int rc_outer = 0; rc_outer < 4; ++rc_outer) {
      for (int rh_outer = 0; rh_outer < 3; ++rh_outer) {
        __syncthreads();
        if ((((int)threadIdx.z) * 17) < (272 - ((int)threadIdx.x))) {
          if (((int)threadIdx.x) < 17) {
            pad_temp_shared[((((int)threadIdx.z) * 17) + ((int)threadIdx.x))] = (((((((1 - ((((((int)threadIdx.z) * 17) + ((int)threadIdx.x)) % 68) / 34)) - rh_outer) <= (((int)blockIdx.y) * 2)) && ((((int)blockIdx.y) * 2) < ((33 - ((((((int)threadIdx.z) * 17) + ((int)threadIdx.x)) % 68) / 34)) - rh_outer))) && (1 <= (((((int)threadIdx.z) * 17) + ((int)threadIdx.x)) % 34))) && ((((((int)threadIdx.z) * 17) + ((int)threadIdx.x)) % 34) < 33)) ? Data[(((((((((((((int)threadIdx.z) * 17) + ((int)threadIdx.x)) / 272) * 16384) + (rc_outer * 4096)) + (((((((int)threadIdx.z) * 17) + ((int)threadIdx.x)) % 272) / 68) * 1024)) + (((int)blockIdx.y) * 64)) + (((((((int)threadIdx.z) * 17) + ((int)threadIdx.x)) % 68) / 34) * 32)) + (rh_outer * 32)) + (((((int)threadIdx.z) * 17) + ((int)threadIdx.x)) % 34)) - 33)] : 0.000000e+00f);
          }
        }
        if (((int)threadIdx.x) < (4 - ((int)threadIdx.z))) {
          if (((int)threadIdx.x) < 1) {
            if ((rc_outer * 4) < ((16 - ((int)threadIdx.z)) - ((int)threadIdx.x))) {
              K0_shared[(((int)threadIdx.x) + ((int)threadIdx.z))] = K0[((((rc_outer * 24) + (((int)threadIdx.x) * 6)) + (((int)threadIdx.z) * 6)) + rr_outer)];
            }
          }
        }
        if (((int)threadIdx.x) < (1 - ((int)threadIdx.z))) {
          if (((int)threadIdx.x) < 1) {
            if (((int)threadIdx.x) < ((3 - rh_outer) - ((int)threadIdx.z))) {
              K1_shared[(((int)threadIdx.x) + ((int)threadIdx.z))] = K1[((((((int)threadIdx.x) * 6) + (((int)threadIdx.z) * 6)) + (rh_outer * 6)) + rr_outer)];
            }
          }
        }
        if (((int)threadIdx.x) < (3 - ((int)threadIdx.z))) {
          if (((int)threadIdx.x) < 1) {
            K2_shared[(((int)threadIdx.x) + ((int)threadIdx.z))] = K2[(((((int)threadIdx.x) * 6) + (((int)threadIdx.z) * 6)) + rr_outer)];
          }
        }
        if (((int)threadIdx.x) < (16 - ((int)threadIdx.z))) {
          if (((int)threadIdx.x) < 1) {
            K3_shared[(((int)threadIdx.x) + ((int)threadIdx.z))] = K3[(((((int)threadIdx.x) * 6) + (((int)threadIdx.z) * 6)) + rr_outer)];
          }
        }
        __syncthreads();
        for (int rc_inner_outer = 0; rc_inner_outer < 4; ++rc_inner_outer) {
          for (int ax2 = 0; ax2 < 2; ++ax2) {
            for (int ax3 = 0; ax3 < 3; ++ax3) {
              pad_temp_shared_local[((ax2 * 3) + ax3)] = pad_temp_shared[((((rc_inner_outer * 68) + (ax2 * 34)) + ax3) + ((int)threadIdx.x))];
            }
          }
          K0_shared_local[0] = K0_shared[rc_inner_outer];
          K1_shared_local[0] = K1_shared[0];
          for (int ax0 = 0; ax0 < 3; ++ax0) {
            K2_shared_local[ax0] = K2_shared[ax0];
          }
          K3_shared_local[0] = K3_shared[((int)threadIdx.z)];
          for (int rw_inner_inner = 0; rw_inner_inner < 3; ++rw_inner_inner) {
            for (int hh_c = 0; hh_c < 2; ++hh_c) {
              Output_local[hh_c] = (Output_local[hh_c] + ((((pad_temp_shared_local[((hh_c * 3) + rw_inner_inner)] * K0_shared_local[0]) * K1_shared_local[0]) * K2_shared_local[rw_inner_inner]) * K3_shared_local[0]));
            }
          }
        }
      }
    }
  }
  for (int hh_inner_inner_inner = 0; hh_inner_inner_inner < 2; ++hh_inner_inner_inner) {
    Output[((((((int)threadIdx.z) * 1024) + (((int)blockIdx.y) * 64)) + (hh_inner_inner_inner * 32)) + ((int)threadIdx.x))] = Output_local[hh_inner_inner_inner];
  }
}


/* void Cp4Conv2dFusedNchwKernelLauncher(const float* U, const float* K0, */
/*     const float* K1, const float* K2, const float* K3, float* V){ */

int main(){

  size_t PROFCOUNT = 1000;

  /* Begin Custom Kernel Profile section */

  float* U;
  float* K0;
  float* K1;
  float* K2;
  float* K3;
  float* V;

  cudaMalloc(&U, (1*16*32*32)*sizeof(float));
  cudaMalloc(&K0, (16*6)*sizeof(float));
  cudaMalloc(&K1, (3*6)*sizeof(float));
  cudaMalloc(&K2, (3*6)*sizeof(float));
  cudaMalloc(&K3, (16*6)*sizeof(float));
  cudaMalloc(&V, (1*16*32*32)*sizeof(float));

  dim3 gridDim0(1, 16, 1);
  dim3 blockDim0(32, 1, 16);

  for (size_t i=0; i<PROFCOUNT; ++i){
    default_function_kernel0<<<gridDim0, blockDim0>>>(U, K0, K1, K2, K3, V);
    cudaDeviceSynchronize();
  }

  cudaFree(U);
  cudaFree(K0);
  cudaFree(K1);
  cudaFree(K2);
  cudaFree(K3);
  cudaFree(V);

}
