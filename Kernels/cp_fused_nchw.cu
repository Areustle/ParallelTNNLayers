#if GOOGLE_CUDA
#define EIGEN_USE_GPU

extern "C" __global__ void default_function_kernel0(const float* __restrict__ Data,
    const float* __restrict__ K0,
    const float* __restrict__ K1,
    const float* __restrict__ K2,
    float* __restrict__ Output) {

   float Output_local[2];
  __shared__ float pad_temp_shared[128];
  __shared__ float K0_shared[12];
  __shared__ float K1_shared[3];
  __shared__ float K2_shared[24];
  for (int hh_inner_outer = 0; hh_inner_outer < 4; ++hh_inner_outer) {
    Output_local[0] = 0.000000e+00f;
    Output_local[1] = 0.000000e+00f;
    for (int rr_outer = 0; rr_outer < 2; ++rr_outer) {
      for (int rc_outer = 0; rc_outer < 4; ++rc_outer) {
        __syncthreads();
        pad_temp_shared[((((int)threadIdx.z) * 16) + ((int)threadIdx.x))] = ((((1 - hh_inner_outer) <= (((int)blockIdx.y) * 4)) && (1 <= (((((int)threadIdx.z) * 16) + ((int)threadIdx.x)) % 32))) ? Data[((((((((((((int)threadIdx.z) * 16) + ((int)threadIdx.x)) / 128) * 16384) + (rc_outer * 4096)) + (((((((int)threadIdx.z) * 16) + ((int)threadIdx.x)) % 128) / 32) * 1024)) + (((int)blockIdx.y) * 128)) + (hh_inner_outer * 32)) + (((((int)threadIdx.z) * 16) + ((int)threadIdx.x)) % 32)) - 33)] : 0.000000e+00f);
        if ((((int)threadIdx.z) * 2) < (12 - ((int)threadIdx.x))) {
          if (((int)threadIdx.x) < 2) {
            if ((rc_outer * 4) < (16 - (((((int)threadIdx.z) * 2) + ((int)threadIdx.x)) / 3))) {
              K0_shared[((((int)threadIdx.z) * 2) + ((int)threadIdx.x))] = K0[((((rc_outer * 24) + ((((((int)threadIdx.z) * 2) + ((int)threadIdx.x)) / 3) * 6)) + (rr_outer * 3)) + (((((int)threadIdx.z) * 2) + ((int)threadIdx.x)) % 3))];
            }
          }
        }
        if (((int)threadIdx.x) < (3 - ((int)threadIdx.z))) {
          if (((int)threadIdx.x) < 1) {
            if (((int)threadIdx.x) < (9 - ((int)threadIdx.z))) {
              K1_shared[(((int)threadIdx.x) + ((int)threadIdx.z))] = K1[(((((((int)threadIdx.x) + ((int)threadIdx.z)) / 3) * 18) + (rr_outer * 3)) + ((((int)threadIdx.x) + ((int)threadIdx.z)) % 3))];
            }
          }
        }
        if ((((int)threadIdx.z) * 3) < (24 - ((int)threadIdx.x))) {
          if (((int)threadIdx.x) < 3) {
            if ((rr_outer * 3) < (6 - (((((int)threadIdx.z) * 3) + ((int)threadIdx.x)) / 8))) {
              K2_shared[((((int)threadIdx.z) * 3) + ((int)threadIdx.x))] = K2[((((rr_outer * 48) + ((((((int)threadIdx.z) * 3) + ((int)threadIdx.x)) / 8) * 16)) + (((int)blockIdx.z) * 8)) + (((((int)threadIdx.z) * 3) + ((int)threadIdx.x)) % 8))];
            }
          }
        }
        __syncthreads();
        Output_local[0] = (Output_local[0] + (((pad_temp_shared[(((int)threadIdx.x) * 2)] * K0_shared[0]) * K1_shared[0]) * K2_shared[((int)threadIdx.z)]));
        Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 1)] * K0_shared[0]) * K1_shared[0]) * K2_shared[((int)threadIdx.z)]));
        Output_local[0] = (Output_local[0] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 32)] * K0_shared[3]) * K1_shared[0]) * K2_shared[((int)threadIdx.z)]));
        Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 33)] * K0_shared[3]) * K1_shared[0]) * K2_shared[((int)threadIdx.z)]));
        Output_local[0] = (Output_local[0] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 64)] * K0_shared[6]) * K1_shared[0]) * K2_shared[((int)threadIdx.z)]));
        Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 65)] * K0_shared[6]) * K1_shared[0]) * K2_shared[((int)threadIdx.z)]));
        Output_local[0] = (Output_local[0] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 96)] * K0_shared[9]) * K1_shared[0]) * K2_shared[((int)threadIdx.z)]));
        Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 97)] * K0_shared[9]) * K1_shared[0]) * K2_shared[((int)threadIdx.z)]));
        Output_local[0] = (Output_local[0] + (((pad_temp_shared[(((int)threadIdx.x) * 2)] * K0_shared[1]) * K1_shared[1]) * K2_shared[(((int)threadIdx.z) + 8)]));
        Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 1)] * K0_shared[1]) * K1_shared[1]) * K2_shared[(((int)threadIdx.z) + 8)]));
        Output_local[0] = (Output_local[0] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 32)] * K0_shared[4]) * K1_shared[1]) * K2_shared[(((int)threadIdx.z) + 8)]));
        Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 33)] * K0_shared[4]) * K1_shared[1]) * K2_shared[(((int)threadIdx.z) + 8)]));
        Output_local[0] = (Output_local[0] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 64)] * K0_shared[7]) * K1_shared[1]) * K2_shared[(((int)threadIdx.z) + 8)]));
        Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 65)] * K0_shared[7]) * K1_shared[1]) * K2_shared[(((int)threadIdx.z) + 8)]));
        Output_local[0] = (Output_local[0] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 96)] * K0_shared[10]) * K1_shared[1]) * K2_shared[(((int)threadIdx.z) + 8)]));
        Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 97)] * K0_shared[10]) * K1_shared[1]) * K2_shared[(((int)threadIdx.z) + 8)]));
        Output_local[0] = (Output_local[0] + (((pad_temp_shared[(((int)threadIdx.x) * 2)] * K0_shared[2]) * K1_shared[2]) * K2_shared[(((int)threadIdx.z) + 16)]));
        Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 1)] * K0_shared[2]) * K1_shared[2]) * K2_shared[(((int)threadIdx.z) + 16)]));
        Output_local[0] = (Output_local[0] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 32)] * K0_shared[5]) * K1_shared[2]) * K2_shared[(((int)threadIdx.z) + 16)]));
        Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 33)] * K0_shared[5]) * K1_shared[2]) * K2_shared[(((int)threadIdx.z) + 16)]));
        Output_local[0] = (Output_local[0] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 64)] * K0_shared[8]) * K1_shared[2]) * K2_shared[(((int)threadIdx.z) + 16)]));
        Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 65)] * K0_shared[8]) * K1_shared[2]) * K2_shared[(((int)threadIdx.z) + 16)]));
        Output_local[0] = (Output_local[0] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 96)] * K0_shared[11]) * K1_shared[2]) * K2_shared[(((int)threadIdx.z) + 16)]));
        Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 97)] * K0_shared[11]) * K1_shared[2]) * K2_shared[(((int)threadIdx.z) + 16)]));
        __syncthreads();
        pad_temp_shared[((((int)threadIdx.z) * 16) + ((int)threadIdx.x))] = (((1 - hh_inner_outer) <= (((int)blockIdx.y) * 4)) ? Data[((((((((((((int)threadIdx.z) * 16) + ((int)threadIdx.x)) / 128) * 16384) + (rc_outer * 4096)) + (((((((int)threadIdx.z) * 16) + ((int)threadIdx.x)) % 128) / 32) * 1024)) + (((int)blockIdx.y) * 128)) + (hh_inner_outer * 32)) + (((((int)threadIdx.z) * 16) + ((int)threadIdx.x)) % 32)) - 32)] : 0.000000e+00f);
        if ((((int)threadIdx.z) * 2) < (12 - ((int)threadIdx.x))) {
          if (((int)threadIdx.x) < 2) {
            if ((rc_outer * 4) < (16 - (((((int)threadIdx.z) * 2) + ((int)threadIdx.x)) / 3))) {
              K0_shared[((((int)threadIdx.z) * 2) + ((int)threadIdx.x))] = K0[((((rc_outer * 24) + ((((((int)threadIdx.z) * 2) + ((int)threadIdx.x)) / 3) * 6)) + (rr_outer * 3)) + (((((int)threadIdx.z) * 2) + ((int)threadIdx.x)) % 3))];
            }
          }
        }
        if (((int)threadIdx.x) < (3 - ((int)threadIdx.z))) {
          if (((int)threadIdx.x) < 1) {
            if (((int)threadIdx.x) < (9 - ((int)threadIdx.z))) {
              K1_shared[(((int)threadIdx.x) + ((int)threadIdx.z))] = K1[((((((((int)threadIdx.x) + ((int)threadIdx.z)) / 3) * 18) + (rr_outer * 3)) + ((((int)threadIdx.x) + ((int)threadIdx.z)) % 3)) + 6)];
            }
          }
        }
        if ((((int)threadIdx.z) * 3) < (24 - ((int)threadIdx.x))) {
          if (((int)threadIdx.x) < 3) {
            if ((rr_outer * 3) < (6 - (((((int)threadIdx.z) * 3) + ((int)threadIdx.x)) / 8))) {
              K2_shared[((((int)threadIdx.z) * 3) + ((int)threadIdx.x))] = K2[((((rr_outer * 48) + ((((((int)threadIdx.z) * 3) + ((int)threadIdx.x)) / 8) * 16)) + (((int)blockIdx.z) * 8)) + (((((int)threadIdx.z) * 3) + ((int)threadIdx.x)) % 8))];
            }
          }
        }
        __syncthreads();
        Output_local[0] = (Output_local[0] + (((pad_temp_shared[(((int)threadIdx.x) * 2)] * K0_shared[0]) * K1_shared[0]) * K2_shared[((int)threadIdx.z)]));
        Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 1)] * K0_shared[0]) * K1_shared[0]) * K2_shared[((int)threadIdx.z)]));
        Output_local[0] = (Output_local[0] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 32)] * K0_shared[3]) * K1_shared[0]) * K2_shared[((int)threadIdx.z)]));
        Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 33)] * K0_shared[3]) * K1_shared[0]) * K2_shared[((int)threadIdx.z)]));
        Output_local[0] = (Output_local[0] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 64)] * K0_shared[6]) * K1_shared[0]) * K2_shared[((int)threadIdx.z)]));
        Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 65)] * K0_shared[6]) * K1_shared[0]) * K2_shared[((int)threadIdx.z)]));
        Output_local[0] = (Output_local[0] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 96)] * K0_shared[9]) * K1_shared[0]) * K2_shared[((int)threadIdx.z)]));
        Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 97)] * K0_shared[9]) * K1_shared[0]) * K2_shared[((int)threadIdx.z)]));
        Output_local[0] = (Output_local[0] + (((pad_temp_shared[(((int)threadIdx.x) * 2)] * K0_shared[1]) * K1_shared[1]) * K2_shared[(((int)threadIdx.z) + 8)]));
        Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 1)] * K0_shared[1]) * K1_shared[1]) * K2_shared[(((int)threadIdx.z) + 8)]));
        Output_local[0] = (Output_local[0] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 32)] * K0_shared[4]) * K1_shared[1]) * K2_shared[(((int)threadIdx.z) + 8)]));
        Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 33)] * K0_shared[4]) * K1_shared[1]) * K2_shared[(((int)threadIdx.z) + 8)]));
        Output_local[0] = (Output_local[0] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 64)] * K0_shared[7]) * K1_shared[1]) * K2_shared[(((int)threadIdx.z) + 8)]));
        Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 65)] * K0_shared[7]) * K1_shared[1]) * K2_shared[(((int)threadIdx.z) + 8)]));
        Output_local[0] = (Output_local[0] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 96)] * K0_shared[10]) * K1_shared[1]) * K2_shared[(((int)threadIdx.z) + 8)]));
        Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 97)] * K0_shared[10]) * K1_shared[1]) * K2_shared[(((int)threadIdx.z) + 8)]));
        Output_local[0] = (Output_local[0] + (((pad_temp_shared[(((int)threadIdx.x) * 2)] * K0_shared[2]) * K1_shared[2]) * K2_shared[(((int)threadIdx.z) + 16)]));
        Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 1)] * K0_shared[2]) * K1_shared[2]) * K2_shared[(((int)threadIdx.z) + 16)]));
        Output_local[0] = (Output_local[0] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 32)] * K0_shared[5]) * K1_shared[2]) * K2_shared[(((int)threadIdx.z) + 16)]));
        Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 33)] * K0_shared[5]) * K1_shared[2]) * K2_shared[(((int)threadIdx.z) + 16)]));
        Output_local[0] = (Output_local[0] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 64)] * K0_shared[8]) * K1_shared[2]) * K2_shared[(((int)threadIdx.z) + 16)]));
        Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 65)] * K0_shared[8]) * K1_shared[2]) * K2_shared[(((int)threadIdx.z) + 16)]));
        Output_local[0] = (Output_local[0] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 96)] * K0_shared[11]) * K1_shared[2]) * K2_shared[(((int)threadIdx.z) + 16)]));
        Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 97)] * K0_shared[11]) * K1_shared[2]) * K2_shared[(((int)threadIdx.z) + 16)]));
        __syncthreads();
        pad_temp_shared[((((int)threadIdx.z) * 16) + ((int)threadIdx.x))] = ((((1 - hh_inner_outer) <= (((int)blockIdx.y) * 4)) && ((((((int)threadIdx.z) * 16) + ((int)threadIdx.x)) % 32) < 31)) ? Data[((((((((((((int)threadIdx.z) * 16) + ((int)threadIdx.x)) / 128) * 16384) + (rc_outer * 4096)) + (((((((int)threadIdx.z) * 16) + ((int)threadIdx.x)) % 128) / 32) * 1024)) + (((int)blockIdx.y) * 128)) + (hh_inner_outer * 32)) + (((((int)threadIdx.z) * 16) + ((int)threadIdx.x)) % 32)) - 31)] : 0.000000e+00f);
        if ((((int)threadIdx.z) * 2) < (12 - ((int)threadIdx.x))) {
          if (((int)threadIdx.x) < 2) {
            if ((rc_outer * 4) < (16 - (((((int)threadIdx.z) * 2) + ((int)threadIdx.x)) / 3))) {
              K0_shared[((((int)threadIdx.z) * 2) + ((int)threadIdx.x))] = K0[((((rc_outer * 24) + ((((((int)threadIdx.z) * 2) + ((int)threadIdx.x)) / 3) * 6)) + (rr_outer * 3)) + (((((int)threadIdx.z) * 2) + ((int)threadIdx.x)) % 3))];
            }
          }
        }
        if (((int)threadIdx.x) < (3 - ((int)threadIdx.z))) {
          if (((int)threadIdx.x) < 1) {
            if (((int)threadIdx.x) < (9 - ((int)threadIdx.z))) {
              K1_shared[(((int)threadIdx.x) + ((int)threadIdx.z))] = K1[((((((((int)threadIdx.x) + ((int)threadIdx.z)) / 3) * 18) + (rr_outer * 3)) + ((((int)threadIdx.x) + ((int)threadIdx.z)) % 3)) + 12)];
            }
          }
        }
        if ((((int)threadIdx.z) * 3) < (24 - ((int)threadIdx.x))) {
          if (((int)threadIdx.x) < 3) {
            if ((rr_outer * 3) < (6 - (((((int)threadIdx.z) * 3) + ((int)threadIdx.x)) / 8))) {
              K2_shared[((((int)threadIdx.z) * 3) + ((int)threadIdx.x))] = K2[((((rr_outer * 48) + ((((((int)threadIdx.z) * 3) + ((int)threadIdx.x)) / 8) * 16)) + (((int)blockIdx.z) * 8)) + (((((int)threadIdx.z) * 3) + ((int)threadIdx.x)) % 8))];
            }
          }
        }
        __syncthreads();
        Output_local[0] = (Output_local[0] + (((pad_temp_shared[(((int)threadIdx.x) * 2)] * K0_shared[0]) * K1_shared[0]) * K2_shared[((int)threadIdx.z)]));
        Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 1)] * K0_shared[0]) * K1_shared[0]) * K2_shared[((int)threadIdx.z)]));
        Output_local[0] = (Output_local[0] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 32)] * K0_shared[3]) * K1_shared[0]) * K2_shared[((int)threadIdx.z)]));
        Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 33)] * K0_shared[3]) * K1_shared[0]) * K2_shared[((int)threadIdx.z)]));
        Output_local[0] = (Output_local[0] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 64)] * K0_shared[6]) * K1_shared[0]) * K2_shared[((int)threadIdx.z)]));
        Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 65)] * K0_shared[6]) * K1_shared[0]) * K2_shared[((int)threadIdx.z)]));
        Output_local[0] = (Output_local[0] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 96)] * K0_shared[9]) * K1_shared[0]) * K2_shared[((int)threadIdx.z)]));
        Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 97)] * K0_shared[9]) * K1_shared[0]) * K2_shared[((int)threadIdx.z)]));
        Output_local[0] = (Output_local[0] + (((pad_temp_shared[(((int)threadIdx.x) * 2)] * K0_shared[1]) * K1_shared[1]) * K2_shared[(((int)threadIdx.z) + 8)]));
        Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 1)] * K0_shared[1]) * K1_shared[1]) * K2_shared[(((int)threadIdx.z) + 8)]));
        Output_local[0] = (Output_local[0] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 32)] * K0_shared[4]) * K1_shared[1]) * K2_shared[(((int)threadIdx.z) + 8)]));
        Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 33)] * K0_shared[4]) * K1_shared[1]) * K2_shared[(((int)threadIdx.z) + 8)]));
        Output_local[0] = (Output_local[0] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 64)] * K0_shared[7]) * K1_shared[1]) * K2_shared[(((int)threadIdx.z) + 8)]));
        Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 65)] * K0_shared[7]) * K1_shared[1]) * K2_shared[(((int)threadIdx.z) + 8)]));
        Output_local[0] = (Output_local[0] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 96)] * K0_shared[10]) * K1_shared[1]) * K2_shared[(((int)threadIdx.z) + 8)]));
        Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 97)] * K0_shared[10]) * K1_shared[1]) * K2_shared[(((int)threadIdx.z) + 8)]));
        Output_local[0] = (Output_local[0] + (((pad_temp_shared[(((int)threadIdx.x) * 2)] * K0_shared[2]) * K1_shared[2]) * K2_shared[(((int)threadIdx.z) + 16)]));
        Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 1)] * K0_shared[2]) * K1_shared[2]) * K2_shared[(((int)threadIdx.z) + 16)]));
        Output_local[0] = (Output_local[0] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 32)] * K0_shared[5]) * K1_shared[2]) * K2_shared[(((int)threadIdx.z) + 16)]));
        Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 33)] * K0_shared[5]) * K1_shared[2]) * K2_shared[(((int)threadIdx.z) + 16)]));
        Output_local[0] = (Output_local[0] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 64)] * K0_shared[8]) * K1_shared[2]) * K2_shared[(((int)threadIdx.z) + 16)]));
        Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 65)] * K0_shared[8]) * K1_shared[2]) * K2_shared[(((int)threadIdx.z) + 16)]));
        Output_local[0] = (Output_local[0] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 96)] * K0_shared[11]) * K1_shared[2]) * K2_shared[(((int)threadIdx.z) + 16)]));
        Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 97)] * K0_shared[11]) * K1_shared[2]) * K2_shared[(((int)threadIdx.z) + 16)]));
        __syncthreads();
        pad_temp_shared[((((int)threadIdx.z) * 16) + ((int)threadIdx.x))] = ((1 <= (((((int)threadIdx.z) * 16) + ((int)threadIdx.x)) % 32)) ? Data[((((((((((((int)threadIdx.z) * 16) + ((int)threadIdx.x)) / 128) * 16384) + (rc_outer * 4096)) + (((((((int)threadIdx.z) * 16) + ((int)threadIdx.x)) % 128) / 32) * 1024)) + (((int)blockIdx.y) * 128)) + (hh_inner_outer * 32)) + (((((int)threadIdx.z) * 16) + ((int)threadIdx.x)) % 32)) - 1)] : 0.000000e+00f);
        if ((((int)threadIdx.z) * 2) < (12 - ((int)threadIdx.x))) {
          if (((int)threadIdx.x) < 2) {
            if ((rc_outer * 4) < (16 - (((((int)threadIdx.z) * 2) + ((int)threadIdx.x)) / 3))) {
              K0_shared[((((int)threadIdx.z) * 2) + ((int)threadIdx.x))] = K0[((((rc_outer * 24) + ((((((int)threadIdx.z) * 2) + ((int)threadIdx.x)) / 3) * 6)) + (rr_outer * 3)) + (((((int)threadIdx.z) * 2) + ((int)threadIdx.x)) % 3))];
            }
          }
        }
        if (((int)threadIdx.x) < (3 - ((int)threadIdx.z))) {
          if (((int)threadIdx.x) < 1) {
            if (((int)threadIdx.x) < (6 - ((int)threadIdx.z))) {
              K1_shared[(((int)threadIdx.x) + ((int)threadIdx.z))] = K1[((((((((int)threadIdx.x) + ((int)threadIdx.z)) / 3) * 18) + (rr_outer * 3)) + ((((int)threadIdx.x) + ((int)threadIdx.z)) % 3)) + 18)];
            }
          }
        }
        if ((((int)threadIdx.z) * 3) < (24 - ((int)threadIdx.x))) {
          if (((int)threadIdx.x) < 3) {
            if ((rr_outer * 3) < (6 - (((((int)threadIdx.z) * 3) + ((int)threadIdx.x)) / 8))) {
              K2_shared[((((int)threadIdx.z) * 3) + ((int)threadIdx.x))] = K2[((((rr_outer * 48) + ((((((int)threadIdx.z) * 3) + ((int)threadIdx.x)) / 8) * 16)) + (((int)blockIdx.z) * 8)) + (((((int)threadIdx.z) * 3) + ((int)threadIdx.x)) % 8))];
            }
          }
        }
        __syncthreads();
        Output_local[0] = (Output_local[0] + (((pad_temp_shared[(((int)threadIdx.x) * 2)] * K0_shared[0]) * K1_shared[0]) * K2_shared[((int)threadIdx.z)]));
        Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 1)] * K0_shared[0]) * K1_shared[0]) * K2_shared[((int)threadIdx.z)]));
        Output_local[0] = (Output_local[0] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 32)] * K0_shared[3]) * K1_shared[0]) * K2_shared[((int)threadIdx.z)]));
        Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 33)] * K0_shared[3]) * K1_shared[0]) * K2_shared[((int)threadIdx.z)]));
        Output_local[0] = (Output_local[0] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 64)] * K0_shared[6]) * K1_shared[0]) * K2_shared[((int)threadIdx.z)]));
        Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 65)] * K0_shared[6]) * K1_shared[0]) * K2_shared[((int)threadIdx.z)]));
        Output_local[0] = (Output_local[0] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 96)] * K0_shared[9]) * K1_shared[0]) * K2_shared[((int)threadIdx.z)]));
        Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 97)] * K0_shared[9]) * K1_shared[0]) * K2_shared[((int)threadIdx.z)]));
        Output_local[0] = (Output_local[0] + (((pad_temp_shared[(((int)threadIdx.x) * 2)] * K0_shared[1]) * K1_shared[1]) * K2_shared[(((int)threadIdx.z) + 8)]));
        Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 1)] * K0_shared[1]) * K1_shared[1]) * K2_shared[(((int)threadIdx.z) + 8)]));
        Output_local[0] = (Output_local[0] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 32)] * K0_shared[4]) * K1_shared[1]) * K2_shared[(((int)threadIdx.z) + 8)]));
        Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 33)] * K0_shared[4]) * K1_shared[1]) * K2_shared[(((int)threadIdx.z) + 8)]));
        Output_local[0] = (Output_local[0] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 64)] * K0_shared[7]) * K1_shared[1]) * K2_shared[(((int)threadIdx.z) + 8)]));
        Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 65)] * K0_shared[7]) * K1_shared[1]) * K2_shared[(((int)threadIdx.z) + 8)]));
        Output_local[0] = (Output_local[0] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 96)] * K0_shared[10]) * K1_shared[1]) * K2_shared[(((int)threadIdx.z) + 8)]));
        Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 97)] * K0_shared[10]) * K1_shared[1]) * K2_shared[(((int)threadIdx.z) + 8)]));
        Output_local[0] = (Output_local[0] + (((pad_temp_shared[(((int)threadIdx.x) * 2)] * K0_shared[2]) * K1_shared[2]) * K2_shared[(((int)threadIdx.z) + 16)]));
        Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 1)] * K0_shared[2]) * K1_shared[2]) * K2_shared[(((int)threadIdx.z) + 16)]));
        Output_local[0] = (Output_local[0] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 32)] * K0_shared[5]) * K1_shared[2]) * K2_shared[(((int)threadIdx.z) + 16)]));
        Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 33)] * K0_shared[5]) * K1_shared[2]) * K2_shared[(((int)threadIdx.z) + 16)]));
        Output_local[0] = (Output_local[0] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 64)] * K0_shared[8]) * K1_shared[2]) * K2_shared[(((int)threadIdx.z) + 16)]));
        Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 65)] * K0_shared[8]) * K1_shared[2]) * K2_shared[(((int)threadIdx.z) + 16)]));
        Output_local[0] = (Output_local[0] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 96)] * K0_shared[11]) * K1_shared[2]) * K2_shared[(((int)threadIdx.z) + 16)]));
        Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 97)] * K0_shared[11]) * K1_shared[2]) * K2_shared[(((int)threadIdx.z) + 16)]));
        __syncthreads();
        pad_temp_shared[((((int)threadIdx.z) * 16) + ((int)threadIdx.x))] = ((bool)1 ? Data[(((((((((((int)threadIdx.z) * 16) + ((int)threadIdx.x)) / 128) * 16384) + (rc_outer * 4096)) + (((((((int)threadIdx.z) * 16) + ((int)threadIdx.x)) % 128) / 32) * 1024)) + (((int)blockIdx.y) * 128)) + (hh_inner_outer * 32)) + (((((int)threadIdx.z) * 16) + ((int)threadIdx.x)) % 32))] : 0.000000e+00f);
        if ((((int)threadIdx.z) * 2) < (12 - ((int)threadIdx.x))) {
          if (((int)threadIdx.x) < 2) {
            if ((rc_outer * 4) < (16 - (((((int)threadIdx.z) * 2) + ((int)threadIdx.x)) / 3))) {
              K0_shared[((((int)threadIdx.z) * 2) + ((int)threadIdx.x))] = K0[((((rc_outer * 24) + ((((((int)threadIdx.z) * 2) + ((int)threadIdx.x)) / 3) * 6)) + (rr_outer * 3)) + (((((int)threadIdx.z) * 2) + ((int)threadIdx.x)) % 3))];
            }
          }
        }
        if (((int)threadIdx.x) < (3 - ((int)threadIdx.z))) {
          if (((int)threadIdx.x) < 1) {
            if (((int)threadIdx.x) < (6 - ((int)threadIdx.z))) {
              K1_shared[(((int)threadIdx.x) + ((int)threadIdx.z))] = K1[((((((((int)threadIdx.x) + ((int)threadIdx.z)) / 3) * 18) + (rr_outer * 3)) + ((((int)threadIdx.x) + ((int)threadIdx.z)) % 3)) + 24)];
            }
          }
        }
        if ((((int)threadIdx.z) * 3) < (24 - ((int)threadIdx.x))) {
          if (((int)threadIdx.x) < 3) {
            if ((rr_outer * 3) < (6 - (((((int)threadIdx.z) * 3) + ((int)threadIdx.x)) / 8))) {
              K2_shared[((((int)threadIdx.z) * 3) + ((int)threadIdx.x))] = K2[((((rr_outer * 48) + ((((((int)threadIdx.z) * 3) + ((int)threadIdx.x)) / 8) * 16)) + (((int)blockIdx.z) * 8)) + (((((int)threadIdx.z) * 3) + ((int)threadIdx.x)) % 8))];
            }
          }
        }
        __syncthreads();
        Output_local[0] = (Output_local[0] + (((pad_temp_shared[(((int)threadIdx.x) * 2)] * K0_shared[0]) * K1_shared[0]) * K2_shared[((int)threadIdx.z)]));
        Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 1)] * K0_shared[0]) * K1_shared[0]) * K2_shared[((int)threadIdx.z)]));
        Output_local[0] = (Output_local[0] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 32)] * K0_shared[3]) * K1_shared[0]) * K2_shared[((int)threadIdx.z)]));
        Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 33)] * K0_shared[3]) * K1_shared[0]) * K2_shared[((int)threadIdx.z)]));
        Output_local[0] = (Output_local[0] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 64)] * K0_shared[6]) * K1_shared[0]) * K2_shared[((int)threadIdx.z)]));
        Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 65)] * K0_shared[6]) * K1_shared[0]) * K2_shared[((int)threadIdx.z)]));
        Output_local[0] = (Output_local[0] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 96)] * K0_shared[9]) * K1_shared[0]) * K2_shared[((int)threadIdx.z)]));
        Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 97)] * K0_shared[9]) * K1_shared[0]) * K2_shared[((int)threadIdx.z)]));
        Output_local[0] = (Output_local[0] + (((pad_temp_shared[(((int)threadIdx.x) * 2)] * K0_shared[1]) * K1_shared[1]) * K2_shared[(((int)threadIdx.z) + 8)]));
        Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 1)] * K0_shared[1]) * K1_shared[1]) * K2_shared[(((int)threadIdx.z) + 8)]));
        Output_local[0] = (Output_local[0] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 32)] * K0_shared[4]) * K1_shared[1]) * K2_shared[(((int)threadIdx.z) + 8)]));
        Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 33)] * K0_shared[4]) * K1_shared[1]) * K2_shared[(((int)threadIdx.z) + 8)]));
        Output_local[0] = (Output_local[0] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 64)] * K0_shared[7]) * K1_shared[1]) * K2_shared[(((int)threadIdx.z) + 8)]));
        Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 65)] * K0_shared[7]) * K1_shared[1]) * K2_shared[(((int)threadIdx.z) + 8)]));
        Output_local[0] = (Output_local[0] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 96)] * K0_shared[10]) * K1_shared[1]) * K2_shared[(((int)threadIdx.z) + 8)]));
        Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 97)] * K0_shared[10]) * K1_shared[1]) * K2_shared[(((int)threadIdx.z) + 8)]));
        Output_local[0] = (Output_local[0] + (((pad_temp_shared[(((int)threadIdx.x) * 2)] * K0_shared[2]) * K1_shared[2]) * K2_shared[(((int)threadIdx.z) + 16)]));
        Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 1)] * K0_shared[2]) * K1_shared[2]) * K2_shared[(((int)threadIdx.z) + 16)]));
        Output_local[0] = (Output_local[0] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 32)] * K0_shared[5]) * K1_shared[2]) * K2_shared[(((int)threadIdx.z) + 16)]));
        Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 33)] * K0_shared[5]) * K1_shared[2]) * K2_shared[(((int)threadIdx.z) + 16)]));
        Output_local[0] = (Output_local[0] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 64)] * K0_shared[8]) * K1_shared[2]) * K2_shared[(((int)threadIdx.z) + 16)]));
        Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 65)] * K0_shared[8]) * K1_shared[2]) * K2_shared[(((int)threadIdx.z) + 16)]));
        Output_local[0] = (Output_local[0] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 96)] * K0_shared[11]) * K1_shared[2]) * K2_shared[(((int)threadIdx.z) + 16)]));
        Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 97)] * K0_shared[11]) * K1_shared[2]) * K2_shared[(((int)threadIdx.z) + 16)]));
        __syncthreads();
        pad_temp_shared[((((int)threadIdx.z) * 16) + ((int)threadIdx.x))] = (((((((int)threadIdx.z) * 16) + ((int)threadIdx.x)) % 32) < 31) ? Data[((((((((((((int)threadIdx.z) * 16) + ((int)threadIdx.x)) / 128) * 16384) + (rc_outer * 4096)) + (((((((int)threadIdx.z) * 16) + ((int)threadIdx.x)) % 128) / 32) * 1024)) + (((int)blockIdx.y) * 128)) + (hh_inner_outer * 32)) + (((((int)threadIdx.z) * 16) + ((int)threadIdx.x)) % 32)) + 1)] : 0.000000e+00f);
        if ((((int)threadIdx.z) * 2) < (12 - ((int)threadIdx.x))) {
          if (((int)threadIdx.x) < 2) {
            if ((rc_outer * 4) < (16 - (((((int)threadIdx.z) * 2) + ((int)threadIdx.x)) / 3))) {
              K0_shared[((((int)threadIdx.z) * 2) + ((int)threadIdx.x))] = K0[((((rc_outer * 24) + ((((((int)threadIdx.z) * 2) + ((int)threadIdx.x)) / 3) * 6)) + (rr_outer * 3)) + (((((int)threadIdx.z) * 2) + ((int)threadIdx.x)) % 3))];
            }
          }
        }
        if (((int)threadIdx.x) < (3 - ((int)threadIdx.z))) {
          if (((int)threadIdx.x) < 1) {
            if (((int)threadIdx.x) < (6 - ((int)threadIdx.z))) {
              K1_shared[(((int)threadIdx.x) + ((int)threadIdx.z))] = K1[((((((((int)threadIdx.x) + ((int)threadIdx.z)) / 3) * 18) + (rr_outer * 3)) + ((((int)threadIdx.x) + ((int)threadIdx.z)) % 3)) + 30)];
            }
          }
        }
        if ((((int)threadIdx.z) * 3) < (24 - ((int)threadIdx.x))) {
          if (((int)threadIdx.x) < 3) {
            if ((rr_outer * 3) < (6 - (((((int)threadIdx.z) * 3) + ((int)threadIdx.x)) / 8))) {
              K2_shared[((((int)threadIdx.z) * 3) + ((int)threadIdx.x))] = K2[((((rr_outer * 48) + ((((((int)threadIdx.z) * 3) + ((int)threadIdx.x)) / 8) * 16)) + (((int)blockIdx.z) * 8)) + (((((int)threadIdx.z) * 3) + ((int)threadIdx.x)) % 8))];
            }
          }
        }
        __syncthreads();
        Output_local[0] = (Output_local[0] + (((pad_temp_shared[(((int)threadIdx.x) * 2)] * K0_shared[0]) * K1_shared[0]) * K2_shared[((int)threadIdx.z)]));
        Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 1)] * K0_shared[0]) * K1_shared[0]) * K2_shared[((int)threadIdx.z)]));
        Output_local[0] = (Output_local[0] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 32)] * K0_shared[3]) * K1_shared[0]) * K2_shared[((int)threadIdx.z)]));
        Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 33)] * K0_shared[3]) * K1_shared[0]) * K2_shared[((int)threadIdx.z)]));
        Output_local[0] = (Output_local[0] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 64)] * K0_shared[6]) * K1_shared[0]) * K2_shared[((int)threadIdx.z)]));
        Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 65)] * K0_shared[6]) * K1_shared[0]) * K2_shared[((int)threadIdx.z)]));
        Output_local[0] = (Output_local[0] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 96)] * K0_shared[9]) * K1_shared[0]) * K2_shared[((int)threadIdx.z)]));
        Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 97)] * K0_shared[9]) * K1_shared[0]) * K2_shared[((int)threadIdx.z)]));
        Output_local[0] = (Output_local[0] + (((pad_temp_shared[(((int)threadIdx.x) * 2)] * K0_shared[1]) * K1_shared[1]) * K2_shared[(((int)threadIdx.z) + 8)]));
        Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 1)] * K0_shared[1]) * K1_shared[1]) * K2_shared[(((int)threadIdx.z) + 8)]));
        Output_local[0] = (Output_local[0] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 32)] * K0_shared[4]) * K1_shared[1]) * K2_shared[(((int)threadIdx.z) + 8)]));
        Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 33)] * K0_shared[4]) * K1_shared[1]) * K2_shared[(((int)threadIdx.z) + 8)]));
        Output_local[0] = (Output_local[0] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 64)] * K0_shared[7]) * K1_shared[1]) * K2_shared[(((int)threadIdx.z) + 8)]));
        Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 65)] * K0_shared[7]) * K1_shared[1]) * K2_shared[(((int)threadIdx.z) + 8)]));
        Output_local[0] = (Output_local[0] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 96)] * K0_shared[10]) * K1_shared[1]) * K2_shared[(((int)threadIdx.z) + 8)]));
        Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 97)] * K0_shared[10]) * K1_shared[1]) * K2_shared[(((int)threadIdx.z) + 8)]));
        Output_local[0] = (Output_local[0] + (((pad_temp_shared[(((int)threadIdx.x) * 2)] * K0_shared[2]) * K1_shared[2]) * K2_shared[(((int)threadIdx.z) + 16)]));
        Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 1)] * K0_shared[2]) * K1_shared[2]) * K2_shared[(((int)threadIdx.z) + 16)]));
        Output_local[0] = (Output_local[0] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 32)] * K0_shared[5]) * K1_shared[2]) * K2_shared[(((int)threadIdx.z) + 16)]));
        Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 33)] * K0_shared[5]) * K1_shared[2]) * K2_shared[(((int)threadIdx.z) + 16)]));
        Output_local[0] = (Output_local[0] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 64)] * K0_shared[8]) * K1_shared[2]) * K2_shared[(((int)threadIdx.z) + 16)]));
        Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 65)] * K0_shared[8]) * K1_shared[2]) * K2_shared[(((int)threadIdx.z) + 16)]));
        Output_local[0] = (Output_local[0] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 96)] * K0_shared[11]) * K1_shared[2]) * K2_shared[(((int)threadIdx.z) + 16)]));
        Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 97)] * K0_shared[11]) * K1_shared[2]) * K2_shared[(((int)threadIdx.z) + 16)]));
        __syncthreads();
        pad_temp_shared[((((int)threadIdx.z) * 16) + ((int)threadIdx.x))] = ((((((int)blockIdx.y) * 4) < (31 - hh_inner_outer)) && (1 <= (((((int)threadIdx.z) * 16) + ((int)threadIdx.x)) % 32))) ? Data[((((((((((((int)threadIdx.z) * 16) + ((int)threadIdx.x)) / 128) * 16384) + (rc_outer * 4096)) + (((((((int)threadIdx.z) * 16) + ((int)threadIdx.x)) % 128) / 32) * 1024)) + (((int)blockIdx.y) * 128)) + (hh_inner_outer * 32)) + (((((int)threadIdx.z) * 16) + ((int)threadIdx.x)) % 32)) + 31)] : 0.000000e+00f);
        if ((((int)threadIdx.z) * 2) < (12 - ((int)threadIdx.x))) {
          if (((int)threadIdx.x) < 2) {
            if ((rc_outer * 4) < (16 - (((((int)threadIdx.z) * 2) + ((int)threadIdx.x)) / 3))) {
              K0_shared[((((int)threadIdx.z) * 2) + ((int)threadIdx.x))] = K0[((((rc_outer * 24) + ((((((int)threadIdx.z) * 2) + ((int)threadIdx.x)) / 3) * 6)) + (rr_outer * 3)) + (((((int)threadIdx.z) * 2) + ((int)threadIdx.x)) % 3))];
            }
          }
        }
        if (((int)threadIdx.x) < (3 - ((int)threadIdx.z))) {
          if (((int)threadIdx.x) < 1) {
            K1_shared[(((int)threadIdx.x) + ((int)threadIdx.z))] = K1[((((((((int)threadIdx.x) + ((int)threadIdx.z)) / 3) * 18) + (rr_outer * 3)) + ((((int)threadIdx.x) + ((int)threadIdx.z)) % 3)) + 36)];
          }
        }
        if ((((int)threadIdx.z) * 3) < (24 - ((int)threadIdx.x))) {
          if (((int)threadIdx.x) < 3) {
            if ((rr_outer * 3) < (6 - (((((int)threadIdx.z) * 3) + ((int)threadIdx.x)) / 8))) {
              K2_shared[((((int)threadIdx.z) * 3) + ((int)threadIdx.x))] = K2[((((rr_outer * 48) + ((((((int)threadIdx.z) * 3) + ((int)threadIdx.x)) / 8) * 16)) + (((int)blockIdx.z) * 8)) + (((((int)threadIdx.z) * 3) + ((int)threadIdx.x)) % 8))];
            }
          }
        }
        __syncthreads();
        Output_local[0] = (Output_local[0] + (((pad_temp_shared[(((int)threadIdx.x) * 2)] * K0_shared[0]) * K1_shared[0]) * K2_shared[((int)threadIdx.z)]));
        Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 1)] * K0_shared[0]) * K1_shared[0]) * K2_shared[((int)threadIdx.z)]));
        Output_local[0] = (Output_local[0] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 32)] * K0_shared[3]) * K1_shared[0]) * K2_shared[((int)threadIdx.z)]));
        Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 33)] * K0_shared[3]) * K1_shared[0]) * K2_shared[((int)threadIdx.z)]));
        Output_local[0] = (Output_local[0] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 64)] * K0_shared[6]) * K1_shared[0]) * K2_shared[((int)threadIdx.z)]));
        Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 65)] * K0_shared[6]) * K1_shared[0]) * K2_shared[((int)threadIdx.z)]));
        Output_local[0] = (Output_local[0] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 96)] * K0_shared[9]) * K1_shared[0]) * K2_shared[((int)threadIdx.z)]));
        Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 97)] * K0_shared[9]) * K1_shared[0]) * K2_shared[((int)threadIdx.z)]));
        Output_local[0] = (Output_local[0] + (((pad_temp_shared[(((int)threadIdx.x) * 2)] * K0_shared[1]) * K1_shared[1]) * K2_shared[(((int)threadIdx.z) + 8)]));
        Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 1)] * K0_shared[1]) * K1_shared[1]) * K2_shared[(((int)threadIdx.z) + 8)]));
        Output_local[0] = (Output_local[0] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 32)] * K0_shared[4]) * K1_shared[1]) * K2_shared[(((int)threadIdx.z) + 8)]));
        Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 33)] * K0_shared[4]) * K1_shared[1]) * K2_shared[(((int)threadIdx.z) + 8)]));
        Output_local[0] = (Output_local[0] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 64)] * K0_shared[7]) * K1_shared[1]) * K2_shared[(((int)threadIdx.z) + 8)]));
        Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 65)] * K0_shared[7]) * K1_shared[1]) * K2_shared[(((int)threadIdx.z) + 8)]));
        Output_local[0] = (Output_local[0] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 96)] * K0_shared[10]) * K1_shared[1]) * K2_shared[(((int)threadIdx.z) + 8)]));
        Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 97)] * K0_shared[10]) * K1_shared[1]) * K2_shared[(((int)threadIdx.z) + 8)]));
        Output_local[0] = (Output_local[0] + (((pad_temp_shared[(((int)threadIdx.x) * 2)] * K0_shared[2]) * K1_shared[2]) * K2_shared[(((int)threadIdx.z) + 16)]));
        Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 1)] * K0_shared[2]) * K1_shared[2]) * K2_shared[(((int)threadIdx.z) + 16)]));
        Output_local[0] = (Output_local[0] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 32)] * K0_shared[5]) * K1_shared[2]) * K2_shared[(((int)threadIdx.z) + 16)]));
        Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 33)] * K0_shared[5]) * K1_shared[2]) * K2_shared[(((int)threadIdx.z) + 16)]));
        Output_local[0] = (Output_local[0] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 64)] * K0_shared[8]) * K1_shared[2]) * K2_shared[(((int)threadIdx.z) + 16)]));
        Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 65)] * K0_shared[8]) * K1_shared[2]) * K2_shared[(((int)threadIdx.z) + 16)]));
        Output_local[0] = (Output_local[0] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 96)] * K0_shared[11]) * K1_shared[2]) * K2_shared[(((int)threadIdx.z) + 16)]));
        Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 97)] * K0_shared[11]) * K1_shared[2]) * K2_shared[(((int)threadIdx.z) + 16)]));
        __syncthreads();
        pad_temp_shared[((((int)threadIdx.z) * 16) + ((int)threadIdx.x))] = (((((int)blockIdx.y) * 4) < (31 - hh_inner_outer)) ? Data[((((((((((((int)threadIdx.z) * 16) + ((int)threadIdx.x)) / 128) * 16384) + (rc_outer * 4096)) + (((((((int)threadIdx.z) * 16) + ((int)threadIdx.x)) % 128) / 32) * 1024)) + (((int)blockIdx.y) * 128)) + (hh_inner_outer * 32)) + (((((int)threadIdx.z) * 16) + ((int)threadIdx.x)) % 32)) + 32)] : 0.000000e+00f);
        if ((((int)threadIdx.z) * 2) < (12 - ((int)threadIdx.x))) {
          if (((int)threadIdx.x) < 2) {
            if ((rc_outer * 4) < (16 - (((((int)threadIdx.z) * 2) + ((int)threadIdx.x)) / 3))) {
              K0_shared[((((int)threadIdx.z) * 2) + ((int)threadIdx.x))] = K0[((((rc_outer * 24) + ((((((int)threadIdx.z) * 2) + ((int)threadIdx.x)) / 3) * 6)) + (rr_outer * 3)) + (((((int)threadIdx.z) * 2) + ((int)threadIdx.x)) % 3))];
            }
          }
        }
        if (((int)threadIdx.x) < (3 - ((int)threadIdx.z))) {
          if (((int)threadIdx.x) < 1) {
            K1_shared[(((int)threadIdx.x) + ((int)threadIdx.z))] = K1[((((((((int)threadIdx.x) + ((int)threadIdx.z)) / 3) * 18) + (rr_outer * 3)) + ((((int)threadIdx.x) + ((int)threadIdx.z)) % 3)) + 42)];
          }
        }
        if ((((int)threadIdx.z) * 3) < (24 - ((int)threadIdx.x))) {
          if (((int)threadIdx.x) < 3) {
            if ((rr_outer * 3) < (6 - (((((int)threadIdx.z) * 3) + ((int)threadIdx.x)) / 8))) {
              K2_shared[((((int)threadIdx.z) * 3) + ((int)threadIdx.x))] = K2[((((rr_outer * 48) + ((((((int)threadIdx.z) * 3) + ((int)threadIdx.x)) / 8) * 16)) + (((int)blockIdx.z) * 8)) + (((((int)threadIdx.z) * 3) + ((int)threadIdx.x)) % 8))];
            }
          }
        }
        __syncthreads();
        Output_local[0] = (Output_local[0] + (((pad_temp_shared[(((int)threadIdx.x) * 2)] * K0_shared[0]) * K1_shared[0]) * K2_shared[((int)threadIdx.z)]));
        Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 1)] * K0_shared[0]) * K1_shared[0]) * K2_shared[((int)threadIdx.z)]));
        Output_local[0] = (Output_local[0] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 32)] * K0_shared[3]) * K1_shared[0]) * K2_shared[((int)threadIdx.z)]));
        Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 33)] * K0_shared[3]) * K1_shared[0]) * K2_shared[((int)threadIdx.z)]));
        Output_local[0] = (Output_local[0] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 64)] * K0_shared[6]) * K1_shared[0]) * K2_shared[((int)threadIdx.z)]));
        Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 65)] * K0_shared[6]) * K1_shared[0]) * K2_shared[((int)threadIdx.z)]));
        Output_local[0] = (Output_local[0] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 96)] * K0_shared[9]) * K1_shared[0]) * K2_shared[((int)threadIdx.z)]));
        Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 97)] * K0_shared[9]) * K1_shared[0]) * K2_shared[((int)threadIdx.z)]));
        Output_local[0] = (Output_local[0] + (((pad_temp_shared[(((int)threadIdx.x) * 2)] * K0_shared[1]) * K1_shared[1]) * K2_shared[(((int)threadIdx.z) + 8)]));
        Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 1)] * K0_shared[1]) * K1_shared[1]) * K2_shared[(((int)threadIdx.z) + 8)]));
        Output_local[0] = (Output_local[0] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 32)] * K0_shared[4]) * K1_shared[1]) * K2_shared[(((int)threadIdx.z) + 8)]));
        Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 33)] * K0_shared[4]) * K1_shared[1]) * K2_shared[(((int)threadIdx.z) + 8)]));
        Output_local[0] = (Output_local[0] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 64)] * K0_shared[7]) * K1_shared[1]) * K2_shared[(((int)threadIdx.z) + 8)]));
        Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 65)] * K0_shared[7]) * K1_shared[1]) * K2_shared[(((int)threadIdx.z) + 8)]));
        Output_local[0] = (Output_local[0] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 96)] * K0_shared[10]) * K1_shared[1]) * K2_shared[(((int)threadIdx.z) + 8)]));
        Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 97)] * K0_shared[10]) * K1_shared[1]) * K2_shared[(((int)threadIdx.z) + 8)]));
        Output_local[0] = (Output_local[0] + (((pad_temp_shared[(((int)threadIdx.x) * 2)] * K0_shared[2]) * K1_shared[2]) * K2_shared[(((int)threadIdx.z) + 16)]));
        Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 1)] * K0_shared[2]) * K1_shared[2]) * K2_shared[(((int)threadIdx.z) + 16)]));
        Output_local[0] = (Output_local[0] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 32)] * K0_shared[5]) * K1_shared[2]) * K2_shared[(((int)threadIdx.z) + 16)]));
        Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 33)] * K0_shared[5]) * K1_shared[2]) * K2_shared[(((int)threadIdx.z) + 16)]));
        Output_local[0] = (Output_local[0] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 64)] * K0_shared[8]) * K1_shared[2]) * K2_shared[(((int)threadIdx.z) + 16)]));
        Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 65)] * K0_shared[8]) * K1_shared[2]) * K2_shared[(((int)threadIdx.z) + 16)]));
        Output_local[0] = (Output_local[0] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 96)] * K0_shared[11]) * K1_shared[2]) * K2_shared[(((int)threadIdx.z) + 16)]));
        Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 97)] * K0_shared[11]) * K1_shared[2]) * K2_shared[(((int)threadIdx.z) + 16)]));
        __syncthreads();
        pad_temp_shared[((((int)threadIdx.z) * 16) + ((int)threadIdx.x))] = ((((((int)blockIdx.y) * 4) < (31 - hh_inner_outer)) && ((((((int)threadIdx.z) * 16) + ((int)threadIdx.x)) % 32) < 31)) ? Data[((((((((((((int)threadIdx.z) * 16) + ((int)threadIdx.x)) / 128) * 16384) + (rc_outer * 4096)) + (((((((int)threadIdx.z) * 16) + ((int)threadIdx.x)) % 128) / 32) * 1024)) + (((int)blockIdx.y) * 128)) + (hh_inner_outer * 32)) + (((((int)threadIdx.z) * 16) + ((int)threadIdx.x)) % 32)) + 33)] : 0.000000e+00f);
        if ((((int)threadIdx.z) * 2) < (12 - ((int)threadIdx.x))) {
          if (((int)threadIdx.x) < 2) {
            if ((rc_outer * 4) < (16 - (((((int)threadIdx.z) * 2) + ((int)threadIdx.x)) / 3))) {
              K0_shared[((((int)threadIdx.z) * 2) + ((int)threadIdx.x))] = K0[((((rc_outer * 24) + ((((((int)threadIdx.z) * 2) + ((int)threadIdx.x)) / 3) * 6)) + (rr_outer * 3)) + (((((int)threadIdx.z) * 2) + ((int)threadIdx.x)) % 3))];
            }
          }
        }
        if (((int)threadIdx.x) < (3 - ((int)threadIdx.z))) {
          if (((int)threadIdx.x) < 1) {
            K1_shared[(((int)threadIdx.x) + ((int)threadIdx.z))] = K1[((((((((int)threadIdx.x) + ((int)threadIdx.z)) / 3) * 18) + (rr_outer * 3)) + ((((int)threadIdx.x) + ((int)threadIdx.z)) % 3)) + 48)];
          }
        }
        if ((((int)threadIdx.z) * 3) < (24 - ((int)threadIdx.x))) {
          if (((int)threadIdx.x) < 3) {
            if ((rr_outer * 3) < (6 - (((((int)threadIdx.z) * 3) + ((int)threadIdx.x)) / 8))) {
              K2_shared[((((int)threadIdx.z) * 3) + ((int)threadIdx.x))] = K2[((((rr_outer * 48) + ((((((int)threadIdx.z) * 3) + ((int)threadIdx.x)) / 8) * 16)) + (((int)blockIdx.z) * 8)) + (((((int)threadIdx.z) * 3) + ((int)threadIdx.x)) % 8))];
            }
          }
        }
        __syncthreads();
        Output_local[0] = (Output_local[0] + (((pad_temp_shared[(((int)threadIdx.x) * 2)] * K0_shared[0]) * K1_shared[0]) * K2_shared[((int)threadIdx.z)]));
        Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 1)] * K0_shared[0]) * K1_shared[0]) * K2_shared[((int)threadIdx.z)]));
        Output_local[0] = (Output_local[0] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 32)] * K0_shared[3]) * K1_shared[0]) * K2_shared[((int)threadIdx.z)]));
        Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 33)] * K0_shared[3]) * K1_shared[0]) * K2_shared[((int)threadIdx.z)]));
        Output_local[0] = (Output_local[0] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 64)] * K0_shared[6]) * K1_shared[0]) * K2_shared[((int)threadIdx.z)]));
        Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 65)] * K0_shared[6]) * K1_shared[0]) * K2_shared[((int)threadIdx.z)]));
        Output_local[0] = (Output_local[0] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 96)] * K0_shared[9]) * K1_shared[0]) * K2_shared[((int)threadIdx.z)]));
        Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 97)] * K0_shared[9]) * K1_shared[0]) * K2_shared[((int)threadIdx.z)]));
        Output_local[0] = (Output_local[0] + (((pad_temp_shared[(((int)threadIdx.x) * 2)] * K0_shared[1]) * K1_shared[1]) * K2_shared[(((int)threadIdx.z) + 8)]));
        Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 1)] * K0_shared[1]) * K1_shared[1]) * K2_shared[(((int)threadIdx.z) + 8)]));
        Output_local[0] = (Output_local[0] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 32)] * K0_shared[4]) * K1_shared[1]) * K2_shared[(((int)threadIdx.z) + 8)]));
        Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 33)] * K0_shared[4]) * K1_shared[1]) * K2_shared[(((int)threadIdx.z) + 8)]));
        Output_local[0] = (Output_local[0] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 64)] * K0_shared[7]) * K1_shared[1]) * K2_shared[(((int)threadIdx.z) + 8)]));
        Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 65)] * K0_shared[7]) * K1_shared[1]) * K2_shared[(((int)threadIdx.z) + 8)]));
        Output_local[0] = (Output_local[0] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 96)] * K0_shared[10]) * K1_shared[1]) * K2_shared[(((int)threadIdx.z) + 8)]));
        Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 97)] * K0_shared[10]) * K1_shared[1]) * K2_shared[(((int)threadIdx.z) + 8)]));
        Output_local[0] = (Output_local[0] + (((pad_temp_shared[(((int)threadIdx.x) * 2)] * K0_shared[2]) * K1_shared[2]) * K2_shared[(((int)threadIdx.z) + 16)]));
        Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 1)] * K0_shared[2]) * K1_shared[2]) * K2_shared[(((int)threadIdx.z) + 16)]));
        Output_local[0] = (Output_local[0] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 32)] * K0_shared[5]) * K1_shared[2]) * K2_shared[(((int)threadIdx.z) + 16)]));
        Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 33)] * K0_shared[5]) * K1_shared[2]) * K2_shared[(((int)threadIdx.z) + 16)]));
        Output_local[0] = (Output_local[0] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 64)] * K0_shared[8]) * K1_shared[2]) * K2_shared[(((int)threadIdx.z) + 16)]));
        Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 65)] * K0_shared[8]) * K1_shared[2]) * K2_shared[(((int)threadIdx.z) + 16)]));
        Output_local[0] = (Output_local[0] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 96)] * K0_shared[11]) * K1_shared[2]) * K2_shared[(((int)threadIdx.z) + 16)]));
        Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((int)threadIdx.x) * 2) + 97)] * K0_shared[11]) * K1_shared[2]) * K2_shared[(((int)threadIdx.z) + 16)]));
      }
    }
    Output[(((((((int)blockIdx.z) * 8192) + (((int)threadIdx.z) * 1024)) + (((int)blockIdx.y) * 128)) + (hh_inner_outer * 32)) + (((int)threadIdx.x) * 2))] = Output_local[0];
    Output[((((((((int)blockIdx.z) * 8192) + (((int)threadIdx.z) * 1024)) + (((int)blockIdx.y) * 128)) + (hh_inner_outer * 32)) + (((int)threadIdx.x) * 2)) + 1)] = Output_local[1];
  }
}


void Conv2dCpFusedNchwKernelLauncher(const float* U, const float* K0,
    const float* K1, const float* K2, float* V){

  dim3 gridDim0(1, 8, 2);
  dim3 blockDim0(16, 1, 8);

  default_function_kernel0<<<gridDim0, blockDim0>>>(U, K0, K1, K2, V);
  cudaDeviceSynchronize();

}

#endif
