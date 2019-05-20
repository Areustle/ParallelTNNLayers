#if GOOGLE_CUDA
#define EIGEN_USE_GPU

extern "C" __global__ void default_function_kernel0(const float* __restrict__ Data,
    const float* __restrict__ K0,
    const float* __restrict__ K1,
    const float* __restrict__ K2,
    float* __restrict__ Output) {

   float Output_local[8];
  __shared__ float pad_temp_shared[256];
  __shared__ float K0_shared[6];
  __shared__ float K1_shared[6];
  __shared__ float K2_shared[24];
  for (int nn_inner_outer = 0; nn_inner_outer < 4; ++nn_inner_outer) {
    for (int hh_inner_outer = 0; hh_inner_outer < 2; ++hh_inner_outer) {
      for (int ww_inner_outer = 0; ww_inner_outer < 2; ++ww_inner_outer) {
        Output_local[0] = 0.000000e+00f;
        Output_local[1] = 0.000000e+00f;
        Output_local[2] = 0.000000e+00f;
        Output_local[3] = 0.000000e+00f;
        Output_local[4] = 0.000000e+00f;
        Output_local[5] = 0.000000e+00f;
        Output_local[6] = 0.000000e+00f;
        Output_local[7] = 0.000000e+00f;
        for (int rc_outer = 0; rc_outer < 16; ++rc_outer) {
          __syncthreads();
          pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 2))] = (((((1 - (((int)threadIdx.x) / 8)) - (((int)threadIdx.y) * 2)) <= ((((int)blockIdx.y) * 16) + (hh_inner_outer * 8))) && ((1 - ((((int)threadIdx.x) * 2) % 16)) <= (ww_inner_outer * 16))) ? Data[((((((((((nn_inner_outer * 32768) + ((((((int)threadIdx.y) * 16) + ((int)threadIdx.x)) / 64) * 16384)) + (((int)threadIdx.z) * 16384)) + (rc_outer * 1024)) + (((int)blockIdx.y) * 512)) + (hh_inner_outer * 256)) + (((((((int)threadIdx.y) * 32) + (((int)threadIdx.x) * 2)) % 128) / 16) * 32)) + (ww_inner_outer * 16)) + ((((int)threadIdx.x) * 2) % 16)) - 33)] : 0.000000e+00f);
          pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 2)) + 1)] = (((((1 - (((((int)threadIdx.x) * 2) + 1) / 16)) - (((int)threadIdx.y) * 2)) <= ((((int)blockIdx.y) * 16) + (hh_inner_outer * 8))) && ((1 - (((((int)threadIdx.x) * 2) + 1) % 16)) <= (ww_inner_outer * 16))) ? Data[((((((((((nn_inner_outer * 32768) + (((((((int)threadIdx.y) * 32) + (((int)threadIdx.x) * 2)) + 1) / 128) * 16384)) + (((int)threadIdx.z) * 16384)) + (rc_outer * 1024)) + (((int)blockIdx.y) * 512)) + (hh_inner_outer * 256)) + ((((((((int)threadIdx.y) * 32) + (((int)threadIdx.x) * 2)) + 1) % 128) / 16) * 32)) + (ww_inner_outer * 16)) + (((((int)threadIdx.x) * 2) + 1) % 16)) - 33)] : 0.000000e+00f);
          if ((((int)threadIdx.z) * 3) < ((6 - ((int)threadIdx.y)) - ((int)threadIdx.x))) {
            if (((int)threadIdx.x) < (3 - ((int)threadIdx.y))) {
              if (((int)threadIdx.x) < 1) {
                if (((((((int)threadIdx.z) * 3) + ((int)threadIdx.x)) + ((int)threadIdx.y)) / 6) < (16 - rc_outer)) {
                  K0_shared[(((((int)threadIdx.z) * 3) + ((int)threadIdx.x)) + ((int)threadIdx.y))] = K0[((((rc_outer * 6) + (((int)threadIdx.z) * 3)) + ((int)threadIdx.x)) + ((int)threadIdx.y))];
                }
              }
            }
          }
          if ((((int)threadIdx.z) * 3) < ((6 - ((int)threadIdx.y)) - ((int)threadIdx.x))) {
            if (((int)threadIdx.x) < (3 - ((int)threadIdx.y))) {
              if (((int)threadIdx.x) < 1) {
                if ((((int)threadIdx.z) * 3) < ((18 - ((int)threadIdx.y)) - ((int)threadIdx.x))) {
                  K1_shared[(((((int)threadIdx.z) * 3) + ((int)threadIdx.x)) + ((int)threadIdx.y))] = K1[((((((((int)threadIdx.z) * 3) + ((int)threadIdx.x)) + ((int)threadIdx.y)) / 6) * 18) + ((((((int)threadIdx.z) * 3) + ((int)threadIdx.x)) + ((int)threadIdx.y)) % 6))];
                }
              }
            }
          }
          if ((((int)threadIdx.z) * 3) < (6 - (((((int)threadIdx.y) * 3) + ((int)threadIdx.x)) / 4))) {
            if (((((int)threadIdx.z) * 12) + (((int)threadIdx.y) * 3)) < (24 - ((int)threadIdx.x))) {
              if ((((int)threadIdx.y) * 3) < (12 - ((int)threadIdx.x))) {
                if (((int)threadIdx.x) < 3) {
                  K2_shared[(((((int)threadIdx.z) * 12) + (((int)threadIdx.y) * 3)) + ((int)threadIdx.x))] = K2[((((((int)threadIdx.z) * 48) + ((((((int)threadIdx.y) * 3) + ((int)threadIdx.x)) / 4) * 16)) + (((int)blockIdx.z) * 4)) + (((((int)threadIdx.y) * 3) + ((int)threadIdx.x)) % 4))];
                }
              }
            }
          }
          __syncthreads();
          Output_local[0] = (Output_local[0] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[0]) * K1_shared[0]) * K2_shared[0]));
          Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[0]) * K1_shared[0]) * K2_shared[0]));
          Output_local[2] = (Output_local[2] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[0]) * K1_shared[0]) * K2_shared[1]));
          Output_local[3] = (Output_local[3] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[0]) * K1_shared[0]) * K2_shared[1]));
          Output_local[4] = (Output_local[4] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[0]) * K1_shared[0]) * K2_shared[2]));
          Output_local[5] = (Output_local[5] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[0]) * K1_shared[0]) * K2_shared[2]));
          Output_local[6] = (Output_local[6] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[0]) * K1_shared[0]) * K2_shared[3]));
          Output_local[7] = (Output_local[7] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[0]) * K1_shared[0]) * K2_shared[3]));
          Output_local[0] = (Output_local[0] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[1]) * K1_shared[1]) * K2_shared[4]));
          Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[1]) * K1_shared[1]) * K2_shared[4]));
          Output_local[2] = (Output_local[2] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[1]) * K1_shared[1]) * K2_shared[5]));
          Output_local[3] = (Output_local[3] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[1]) * K1_shared[1]) * K2_shared[5]));
          Output_local[4] = (Output_local[4] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[1]) * K1_shared[1]) * K2_shared[6]));
          Output_local[5] = (Output_local[5] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[1]) * K1_shared[1]) * K2_shared[6]));
          Output_local[6] = (Output_local[6] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[1]) * K1_shared[1]) * K2_shared[7]));
          Output_local[7] = (Output_local[7] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[1]) * K1_shared[1]) * K2_shared[7]));
          Output_local[0] = (Output_local[0] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[2]) * K1_shared[2]) * K2_shared[8]));
          Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[2]) * K1_shared[2]) * K2_shared[8]));
          Output_local[2] = (Output_local[2] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[2]) * K1_shared[2]) * K2_shared[9]));
          Output_local[3] = (Output_local[3] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[2]) * K1_shared[2]) * K2_shared[9]));
          Output_local[4] = (Output_local[4] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[2]) * K1_shared[2]) * K2_shared[10]));
          Output_local[5] = (Output_local[5] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[2]) * K1_shared[2]) * K2_shared[10]));
          Output_local[6] = (Output_local[6] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[2]) * K1_shared[2]) * K2_shared[11]));
          Output_local[7] = (Output_local[7] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[2]) * K1_shared[2]) * K2_shared[11]));
          Output_local[0] = (Output_local[0] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[3]) * K1_shared[3]) * K2_shared[12]));
          Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[3]) * K1_shared[3]) * K2_shared[12]));
          Output_local[2] = (Output_local[2] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[3]) * K1_shared[3]) * K2_shared[13]));
          Output_local[3] = (Output_local[3] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[3]) * K1_shared[3]) * K2_shared[13]));
          Output_local[4] = (Output_local[4] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[3]) * K1_shared[3]) * K2_shared[14]));
          Output_local[5] = (Output_local[5] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[3]) * K1_shared[3]) * K2_shared[14]));
          Output_local[6] = (Output_local[6] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[3]) * K1_shared[3]) * K2_shared[15]));
          Output_local[7] = (Output_local[7] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[3]) * K1_shared[3]) * K2_shared[15]));
          Output_local[0] = (Output_local[0] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[4]) * K1_shared[4]) * K2_shared[16]));
          Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[4]) * K1_shared[4]) * K2_shared[16]));
          Output_local[2] = (Output_local[2] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[4]) * K1_shared[4]) * K2_shared[17]));
          Output_local[3] = (Output_local[3] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[4]) * K1_shared[4]) * K2_shared[17]));
          Output_local[4] = (Output_local[4] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[4]) * K1_shared[4]) * K2_shared[18]));
          Output_local[5] = (Output_local[5] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[4]) * K1_shared[4]) * K2_shared[18]));
          Output_local[6] = (Output_local[6] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[4]) * K1_shared[4]) * K2_shared[19]));
          Output_local[7] = (Output_local[7] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[4]) * K1_shared[4]) * K2_shared[19]));
          Output_local[0] = (Output_local[0] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[5]) * K1_shared[5]) * K2_shared[20]));
          Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[5]) * K1_shared[5]) * K2_shared[20]));
          Output_local[2] = (Output_local[2] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[5]) * K1_shared[5]) * K2_shared[21]));
          Output_local[3] = (Output_local[3] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[5]) * K1_shared[5]) * K2_shared[21]));
          Output_local[4] = (Output_local[4] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[5]) * K1_shared[5]) * K2_shared[22]));
          Output_local[5] = (Output_local[5] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[5]) * K1_shared[5]) * K2_shared[22]));
          Output_local[6] = (Output_local[6] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[5]) * K1_shared[5]) * K2_shared[23]));
          Output_local[7] = (Output_local[7] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[5]) * K1_shared[5]) * K2_shared[23]));
          __syncthreads();
          pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 2))] = ((((1 - (((int)threadIdx.x) / 8)) - (((int)threadIdx.y) * 2)) <= ((((int)blockIdx.y) * 16) + (hh_inner_outer * 8))) ? Data[((((((((((nn_inner_outer * 32768) + ((((((int)threadIdx.y) * 16) + ((int)threadIdx.x)) / 64) * 16384)) + (((int)threadIdx.z) * 16384)) + (rc_outer * 1024)) + (((int)blockIdx.y) * 512)) + (hh_inner_outer * 256)) + (((((((int)threadIdx.y) * 32) + (((int)threadIdx.x) * 2)) % 128) / 16) * 32)) + (ww_inner_outer * 16)) + ((((int)threadIdx.x) * 2) % 16)) - 32)] : 0.000000e+00f);
          pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 2)) + 1)] = ((((1 - (((((int)threadIdx.x) * 2) + 1) / 16)) - (((int)threadIdx.y) * 2)) <= ((((int)blockIdx.y) * 16) + (hh_inner_outer * 8))) ? Data[((((((((((nn_inner_outer * 32768) + (((((((int)threadIdx.y) * 32) + (((int)threadIdx.x) * 2)) + 1) / 128) * 16384)) + (((int)threadIdx.z) * 16384)) + (rc_outer * 1024)) + (((int)blockIdx.y) * 512)) + (hh_inner_outer * 256)) + ((((((((int)threadIdx.y) * 32) + (((int)threadIdx.x) * 2)) + 1) % 128) / 16) * 32)) + (ww_inner_outer * 16)) + (((((int)threadIdx.x) * 2) + 1) % 16)) - 32)] : 0.000000e+00f);
          if ((((int)threadIdx.z) * 3) < ((6 - ((int)threadIdx.y)) - ((int)threadIdx.x))) {
            if (((int)threadIdx.x) < (3 - ((int)threadIdx.y))) {
              if (((int)threadIdx.x) < 1) {
                if (((((((int)threadIdx.z) * 3) + ((int)threadIdx.x)) + ((int)threadIdx.y)) / 6) < (16 - rc_outer)) {
                  K0_shared[(((((int)threadIdx.z) * 3) + ((int)threadIdx.x)) + ((int)threadIdx.y))] = K0[((((rc_outer * 6) + (((int)threadIdx.z) * 3)) + ((int)threadIdx.x)) + ((int)threadIdx.y))];
                }
              }
            }
          }
          if ((((int)threadIdx.z) * 3) < ((6 - ((int)threadIdx.y)) - ((int)threadIdx.x))) {
            if (((int)threadIdx.x) < (3 - ((int)threadIdx.y))) {
              if (((int)threadIdx.x) < 1) {
                if ((((int)threadIdx.z) * 3) < ((18 - ((int)threadIdx.y)) - ((int)threadIdx.x))) {
                  K1_shared[(((((int)threadIdx.z) * 3) + ((int)threadIdx.x)) + ((int)threadIdx.y))] = K1[(((((((((int)threadIdx.z) * 3) + ((int)threadIdx.x)) + ((int)threadIdx.y)) / 6) * 18) + ((((((int)threadIdx.z) * 3) + ((int)threadIdx.x)) + ((int)threadIdx.y)) % 6)) + 6)];
                }
              }
            }
          }
          if ((((int)threadIdx.z) * 3) < (6 - (((((int)threadIdx.y) * 3) + ((int)threadIdx.x)) / 4))) {
            if (((((int)threadIdx.z) * 12) + (((int)threadIdx.y) * 3)) < (24 - ((int)threadIdx.x))) {
              if ((((int)threadIdx.y) * 3) < (12 - ((int)threadIdx.x))) {
                if (((int)threadIdx.x) < 3) {
                  K2_shared[(((((int)threadIdx.z) * 12) + (((int)threadIdx.y) * 3)) + ((int)threadIdx.x))] = K2[((((((int)threadIdx.z) * 48) + ((((((int)threadIdx.y) * 3) + ((int)threadIdx.x)) / 4) * 16)) + (((int)blockIdx.z) * 4)) + (((((int)threadIdx.y) * 3) + ((int)threadIdx.x)) % 4))];
                }
              }
            }
          }
          __syncthreads();
          Output_local[0] = (Output_local[0] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[0]) * K1_shared[0]) * K2_shared[0]));
          Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[0]) * K1_shared[0]) * K2_shared[0]));
          Output_local[2] = (Output_local[2] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[0]) * K1_shared[0]) * K2_shared[1]));
          Output_local[3] = (Output_local[3] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[0]) * K1_shared[0]) * K2_shared[1]));
          Output_local[4] = (Output_local[4] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[0]) * K1_shared[0]) * K2_shared[2]));
          Output_local[5] = (Output_local[5] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[0]) * K1_shared[0]) * K2_shared[2]));
          Output_local[6] = (Output_local[6] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[0]) * K1_shared[0]) * K2_shared[3]));
          Output_local[7] = (Output_local[7] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[0]) * K1_shared[0]) * K2_shared[3]));
          Output_local[0] = (Output_local[0] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[1]) * K1_shared[1]) * K2_shared[4]));
          Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[1]) * K1_shared[1]) * K2_shared[4]));
          Output_local[2] = (Output_local[2] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[1]) * K1_shared[1]) * K2_shared[5]));
          Output_local[3] = (Output_local[3] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[1]) * K1_shared[1]) * K2_shared[5]));
          Output_local[4] = (Output_local[4] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[1]) * K1_shared[1]) * K2_shared[6]));
          Output_local[5] = (Output_local[5] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[1]) * K1_shared[1]) * K2_shared[6]));
          Output_local[6] = (Output_local[6] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[1]) * K1_shared[1]) * K2_shared[7]));
          Output_local[7] = (Output_local[7] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[1]) * K1_shared[1]) * K2_shared[7]));
          Output_local[0] = (Output_local[0] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[2]) * K1_shared[2]) * K2_shared[8]));
          Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[2]) * K1_shared[2]) * K2_shared[8]));
          Output_local[2] = (Output_local[2] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[2]) * K1_shared[2]) * K2_shared[9]));
          Output_local[3] = (Output_local[3] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[2]) * K1_shared[2]) * K2_shared[9]));
          Output_local[4] = (Output_local[4] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[2]) * K1_shared[2]) * K2_shared[10]));
          Output_local[5] = (Output_local[5] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[2]) * K1_shared[2]) * K2_shared[10]));
          Output_local[6] = (Output_local[6] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[2]) * K1_shared[2]) * K2_shared[11]));
          Output_local[7] = (Output_local[7] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[2]) * K1_shared[2]) * K2_shared[11]));
          Output_local[0] = (Output_local[0] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[3]) * K1_shared[3]) * K2_shared[12]));
          Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[3]) * K1_shared[3]) * K2_shared[12]));
          Output_local[2] = (Output_local[2] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[3]) * K1_shared[3]) * K2_shared[13]));
          Output_local[3] = (Output_local[3] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[3]) * K1_shared[3]) * K2_shared[13]));
          Output_local[4] = (Output_local[4] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[3]) * K1_shared[3]) * K2_shared[14]));
          Output_local[5] = (Output_local[5] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[3]) * K1_shared[3]) * K2_shared[14]));
          Output_local[6] = (Output_local[6] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[3]) * K1_shared[3]) * K2_shared[15]));
          Output_local[7] = (Output_local[7] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[3]) * K1_shared[3]) * K2_shared[15]));
          Output_local[0] = (Output_local[0] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[4]) * K1_shared[4]) * K2_shared[16]));
          Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[4]) * K1_shared[4]) * K2_shared[16]));
          Output_local[2] = (Output_local[2] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[4]) * K1_shared[4]) * K2_shared[17]));
          Output_local[3] = (Output_local[3] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[4]) * K1_shared[4]) * K2_shared[17]));
          Output_local[4] = (Output_local[4] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[4]) * K1_shared[4]) * K2_shared[18]));
          Output_local[5] = (Output_local[5] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[4]) * K1_shared[4]) * K2_shared[18]));
          Output_local[6] = (Output_local[6] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[4]) * K1_shared[4]) * K2_shared[19]));
          Output_local[7] = (Output_local[7] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[4]) * K1_shared[4]) * K2_shared[19]));
          Output_local[0] = (Output_local[0] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[5]) * K1_shared[5]) * K2_shared[20]));
          Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[5]) * K1_shared[5]) * K2_shared[20]));
          Output_local[2] = (Output_local[2] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[5]) * K1_shared[5]) * K2_shared[21]));
          Output_local[3] = (Output_local[3] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[5]) * K1_shared[5]) * K2_shared[21]));
          Output_local[4] = (Output_local[4] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[5]) * K1_shared[5]) * K2_shared[22]));
          Output_local[5] = (Output_local[5] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[5]) * K1_shared[5]) * K2_shared[22]));
          Output_local[6] = (Output_local[6] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[5]) * K1_shared[5]) * K2_shared[23]));
          Output_local[7] = (Output_local[7] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[5]) * K1_shared[5]) * K2_shared[23]));
          __syncthreads();
          pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 2))] = (((((1 - (((int)threadIdx.x) / 8)) - (((int)threadIdx.y) * 2)) <= ((((int)blockIdx.y) * 16) + (hh_inner_outer * 8))) && ((ww_inner_outer * 16) < (31 - ((((int)threadIdx.x) * 2) % 16)))) ? Data[((((((((((nn_inner_outer * 32768) + ((((((int)threadIdx.y) * 16) + ((int)threadIdx.x)) / 64) * 16384)) + (((int)threadIdx.z) * 16384)) + (rc_outer * 1024)) + (((int)blockIdx.y) * 512)) + (hh_inner_outer * 256)) + (((((((int)threadIdx.y) * 32) + (((int)threadIdx.x) * 2)) % 128) / 16) * 32)) + (ww_inner_outer * 16)) + ((((int)threadIdx.x) * 2) % 16)) - 31)] : 0.000000e+00f);
          pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 2)) + 1)] = (((((1 - (((((int)threadIdx.x) * 2) + 1) / 16)) - (((int)threadIdx.y) * 2)) <= ((((int)blockIdx.y) * 16) + (hh_inner_outer * 8))) && ((ww_inner_outer * 16) < (31 - (((((int)threadIdx.x) * 2) + 1) % 16)))) ? Data[((((((((((nn_inner_outer * 32768) + (((((((int)threadIdx.y) * 32) + (((int)threadIdx.x) * 2)) + 1) / 128) * 16384)) + (((int)threadIdx.z) * 16384)) + (rc_outer * 1024)) + (((int)blockIdx.y) * 512)) + (hh_inner_outer * 256)) + ((((((((int)threadIdx.y) * 32) + (((int)threadIdx.x) * 2)) + 1) % 128) / 16) * 32)) + (ww_inner_outer * 16)) + (((((int)threadIdx.x) * 2) + 1) % 16)) - 31)] : 0.000000e+00f);
          if ((((int)threadIdx.z) * 3) < ((6 - ((int)threadIdx.y)) - ((int)threadIdx.x))) {
            if (((int)threadIdx.x) < (3 - ((int)threadIdx.y))) {
              if (((int)threadIdx.x) < 1) {
                if (((((((int)threadIdx.z) * 3) + ((int)threadIdx.x)) + ((int)threadIdx.y)) / 6) < (16 - rc_outer)) {
                  K0_shared[(((((int)threadIdx.z) * 3) + ((int)threadIdx.x)) + ((int)threadIdx.y))] = K0[((((rc_outer * 6) + (((int)threadIdx.z) * 3)) + ((int)threadIdx.x)) + ((int)threadIdx.y))];
                }
              }
            }
          }
          if ((((int)threadIdx.z) * 3) < ((6 - ((int)threadIdx.y)) - ((int)threadIdx.x))) {
            if (((int)threadIdx.x) < (3 - ((int)threadIdx.y))) {
              if (((int)threadIdx.x) < 1) {
                if ((((int)threadIdx.z) * 3) < ((18 - ((int)threadIdx.y)) - ((int)threadIdx.x))) {
                  K1_shared[(((((int)threadIdx.z) * 3) + ((int)threadIdx.x)) + ((int)threadIdx.y))] = K1[(((((((((int)threadIdx.z) * 3) + ((int)threadIdx.x)) + ((int)threadIdx.y)) / 6) * 18) + ((((((int)threadIdx.z) * 3) + ((int)threadIdx.x)) + ((int)threadIdx.y)) % 6)) + 12)];
                }
              }
            }
          }
          if ((((int)threadIdx.z) * 3) < (6 - (((((int)threadIdx.y) * 3) + ((int)threadIdx.x)) / 4))) {
            if (((((int)threadIdx.z) * 12) + (((int)threadIdx.y) * 3)) < (24 - ((int)threadIdx.x))) {
              if ((((int)threadIdx.y) * 3) < (12 - ((int)threadIdx.x))) {
                if (((int)threadIdx.x) < 3) {
                  K2_shared[(((((int)threadIdx.z) * 12) + (((int)threadIdx.y) * 3)) + ((int)threadIdx.x))] = K2[((((((int)threadIdx.z) * 48) + ((((((int)threadIdx.y) * 3) + ((int)threadIdx.x)) / 4) * 16)) + (((int)blockIdx.z) * 4)) + (((((int)threadIdx.y) * 3) + ((int)threadIdx.x)) % 4))];
                }
              }
            }
          }
          __syncthreads();
          Output_local[0] = (Output_local[0] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[0]) * K1_shared[0]) * K2_shared[0]));
          Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[0]) * K1_shared[0]) * K2_shared[0]));
          Output_local[2] = (Output_local[2] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[0]) * K1_shared[0]) * K2_shared[1]));
          Output_local[3] = (Output_local[3] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[0]) * K1_shared[0]) * K2_shared[1]));
          Output_local[4] = (Output_local[4] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[0]) * K1_shared[0]) * K2_shared[2]));
          Output_local[5] = (Output_local[5] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[0]) * K1_shared[0]) * K2_shared[2]));
          Output_local[6] = (Output_local[6] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[0]) * K1_shared[0]) * K2_shared[3]));
          Output_local[7] = (Output_local[7] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[0]) * K1_shared[0]) * K2_shared[3]));
          Output_local[0] = (Output_local[0] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[1]) * K1_shared[1]) * K2_shared[4]));
          Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[1]) * K1_shared[1]) * K2_shared[4]));
          Output_local[2] = (Output_local[2] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[1]) * K1_shared[1]) * K2_shared[5]));
          Output_local[3] = (Output_local[3] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[1]) * K1_shared[1]) * K2_shared[5]));
          Output_local[4] = (Output_local[4] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[1]) * K1_shared[1]) * K2_shared[6]));
          Output_local[5] = (Output_local[5] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[1]) * K1_shared[1]) * K2_shared[6]));
          Output_local[6] = (Output_local[6] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[1]) * K1_shared[1]) * K2_shared[7]));
          Output_local[7] = (Output_local[7] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[1]) * K1_shared[1]) * K2_shared[7]));
          Output_local[0] = (Output_local[0] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[2]) * K1_shared[2]) * K2_shared[8]));
          Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[2]) * K1_shared[2]) * K2_shared[8]));
          Output_local[2] = (Output_local[2] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[2]) * K1_shared[2]) * K2_shared[9]));
          Output_local[3] = (Output_local[3] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[2]) * K1_shared[2]) * K2_shared[9]));
          Output_local[4] = (Output_local[4] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[2]) * K1_shared[2]) * K2_shared[10]));
          Output_local[5] = (Output_local[5] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[2]) * K1_shared[2]) * K2_shared[10]));
          Output_local[6] = (Output_local[6] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[2]) * K1_shared[2]) * K2_shared[11]));
          Output_local[7] = (Output_local[7] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[2]) * K1_shared[2]) * K2_shared[11]));
          Output_local[0] = (Output_local[0] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[3]) * K1_shared[3]) * K2_shared[12]));
          Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[3]) * K1_shared[3]) * K2_shared[12]));
          Output_local[2] = (Output_local[2] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[3]) * K1_shared[3]) * K2_shared[13]));
          Output_local[3] = (Output_local[3] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[3]) * K1_shared[3]) * K2_shared[13]));
          Output_local[4] = (Output_local[4] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[3]) * K1_shared[3]) * K2_shared[14]));
          Output_local[5] = (Output_local[5] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[3]) * K1_shared[3]) * K2_shared[14]));
          Output_local[6] = (Output_local[6] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[3]) * K1_shared[3]) * K2_shared[15]));
          Output_local[7] = (Output_local[7] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[3]) * K1_shared[3]) * K2_shared[15]));
          Output_local[0] = (Output_local[0] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[4]) * K1_shared[4]) * K2_shared[16]));
          Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[4]) * K1_shared[4]) * K2_shared[16]));
          Output_local[2] = (Output_local[2] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[4]) * K1_shared[4]) * K2_shared[17]));
          Output_local[3] = (Output_local[3] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[4]) * K1_shared[4]) * K2_shared[17]));
          Output_local[4] = (Output_local[4] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[4]) * K1_shared[4]) * K2_shared[18]));
          Output_local[5] = (Output_local[5] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[4]) * K1_shared[4]) * K2_shared[18]));
          Output_local[6] = (Output_local[6] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[4]) * K1_shared[4]) * K2_shared[19]));
          Output_local[7] = (Output_local[7] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[4]) * K1_shared[4]) * K2_shared[19]));
          Output_local[0] = (Output_local[0] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[5]) * K1_shared[5]) * K2_shared[20]));
          Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[5]) * K1_shared[5]) * K2_shared[20]));
          Output_local[2] = (Output_local[2] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[5]) * K1_shared[5]) * K2_shared[21]));
          Output_local[3] = (Output_local[3] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[5]) * K1_shared[5]) * K2_shared[21]));
          Output_local[4] = (Output_local[4] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[5]) * K1_shared[5]) * K2_shared[22]));
          Output_local[5] = (Output_local[5] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[5]) * K1_shared[5]) * K2_shared[22]));
          Output_local[6] = (Output_local[6] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[5]) * K1_shared[5]) * K2_shared[23]));
          Output_local[7] = (Output_local[7] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[5]) * K1_shared[5]) * K2_shared[23]));
          __syncthreads();
          pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 2))] = (((1 - ((((int)threadIdx.x) * 2) % 16)) <= (ww_inner_outer * 16)) ? Data[((((((((((nn_inner_outer * 32768) + ((((((int)threadIdx.y) * 16) + ((int)threadIdx.x)) / 64) * 16384)) + (((int)threadIdx.z) * 16384)) + (rc_outer * 1024)) + (((int)blockIdx.y) * 512)) + (hh_inner_outer * 256)) + (((((((int)threadIdx.y) * 32) + (((int)threadIdx.x) * 2)) % 128) / 16) * 32)) + (ww_inner_outer * 16)) + ((((int)threadIdx.x) * 2) % 16)) - 1)] : 0.000000e+00f);
          pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 2)) + 1)] = (((1 - (((((int)threadIdx.x) * 2) + 1) % 16)) <= (ww_inner_outer * 16)) ? Data[((((((((((nn_inner_outer * 32768) + (((((((int)threadIdx.y) * 32) + (((int)threadIdx.x) * 2)) + 1) / 128) * 16384)) + (((int)threadIdx.z) * 16384)) + (rc_outer * 1024)) + (((int)blockIdx.y) * 512)) + (hh_inner_outer * 256)) + ((((((((int)threadIdx.y) * 32) + (((int)threadIdx.x) * 2)) + 1) % 128) / 16) * 32)) + (ww_inner_outer * 16)) + (((((int)threadIdx.x) * 2) + 1) % 16)) - 1)] : 0.000000e+00f);
          if ((((int)threadIdx.z) * 3) < ((6 - ((int)threadIdx.y)) - ((int)threadIdx.x))) {
            if (((int)threadIdx.x) < (3 - ((int)threadIdx.y))) {
              if (((int)threadIdx.x) < 1) {
                if (((((((int)threadIdx.z) * 3) + ((int)threadIdx.x)) + ((int)threadIdx.y)) / 6) < (16 - rc_outer)) {
                  K0_shared[(((((int)threadIdx.z) * 3) + ((int)threadIdx.x)) + ((int)threadIdx.y))] = K0[((((rc_outer * 6) + (((int)threadIdx.z) * 3)) + ((int)threadIdx.x)) + ((int)threadIdx.y))];
                }
              }
            }
          }
          if ((((int)threadIdx.z) * 3) < ((6 - ((int)threadIdx.y)) - ((int)threadIdx.x))) {
            if (((int)threadIdx.x) < (3 - ((int)threadIdx.y))) {
              if (((int)threadIdx.x) < 1) {
                if ((((int)threadIdx.z) * 3) < ((12 - ((int)threadIdx.y)) - ((int)threadIdx.x))) {
                  K1_shared[(((((int)threadIdx.z) * 3) + ((int)threadIdx.x)) + ((int)threadIdx.y))] = K1[(((((((((int)threadIdx.z) * 3) + ((int)threadIdx.x)) + ((int)threadIdx.y)) / 6) * 18) + ((((((int)threadIdx.z) * 3) + ((int)threadIdx.x)) + ((int)threadIdx.y)) % 6)) + 18)];
                }
              }
            }
          }
          if ((((int)threadIdx.z) * 3) < (6 - (((((int)threadIdx.y) * 3) + ((int)threadIdx.x)) / 4))) {
            if (((((int)threadIdx.z) * 12) + (((int)threadIdx.y) * 3)) < (24 - ((int)threadIdx.x))) {
              if ((((int)threadIdx.y) * 3) < (12 - ((int)threadIdx.x))) {
                if (((int)threadIdx.x) < 3) {
                  K2_shared[(((((int)threadIdx.z) * 12) + (((int)threadIdx.y) * 3)) + ((int)threadIdx.x))] = K2[((((((int)threadIdx.z) * 48) + ((((((int)threadIdx.y) * 3) + ((int)threadIdx.x)) / 4) * 16)) + (((int)blockIdx.z) * 4)) + (((((int)threadIdx.y) * 3) + ((int)threadIdx.x)) % 4))];
                }
              }
            }
          }
          __syncthreads();
          Output_local[0] = (Output_local[0] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[0]) * K1_shared[0]) * K2_shared[0]));
          Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[0]) * K1_shared[0]) * K2_shared[0]));
          Output_local[2] = (Output_local[2] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[0]) * K1_shared[0]) * K2_shared[1]));
          Output_local[3] = (Output_local[3] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[0]) * K1_shared[0]) * K2_shared[1]));
          Output_local[4] = (Output_local[4] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[0]) * K1_shared[0]) * K2_shared[2]));
          Output_local[5] = (Output_local[5] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[0]) * K1_shared[0]) * K2_shared[2]));
          Output_local[6] = (Output_local[6] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[0]) * K1_shared[0]) * K2_shared[3]));
          Output_local[7] = (Output_local[7] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[0]) * K1_shared[0]) * K2_shared[3]));
          Output_local[0] = (Output_local[0] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[1]) * K1_shared[1]) * K2_shared[4]));
          Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[1]) * K1_shared[1]) * K2_shared[4]));
          Output_local[2] = (Output_local[2] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[1]) * K1_shared[1]) * K2_shared[5]));
          Output_local[3] = (Output_local[3] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[1]) * K1_shared[1]) * K2_shared[5]));
          Output_local[4] = (Output_local[4] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[1]) * K1_shared[1]) * K2_shared[6]));
          Output_local[5] = (Output_local[5] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[1]) * K1_shared[1]) * K2_shared[6]));
          Output_local[6] = (Output_local[6] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[1]) * K1_shared[1]) * K2_shared[7]));
          Output_local[7] = (Output_local[7] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[1]) * K1_shared[1]) * K2_shared[7]));
          Output_local[0] = (Output_local[0] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[2]) * K1_shared[2]) * K2_shared[8]));
          Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[2]) * K1_shared[2]) * K2_shared[8]));
          Output_local[2] = (Output_local[2] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[2]) * K1_shared[2]) * K2_shared[9]));
          Output_local[3] = (Output_local[3] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[2]) * K1_shared[2]) * K2_shared[9]));
          Output_local[4] = (Output_local[4] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[2]) * K1_shared[2]) * K2_shared[10]));
          Output_local[5] = (Output_local[5] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[2]) * K1_shared[2]) * K2_shared[10]));
          Output_local[6] = (Output_local[6] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[2]) * K1_shared[2]) * K2_shared[11]));
          Output_local[7] = (Output_local[7] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[2]) * K1_shared[2]) * K2_shared[11]));
          Output_local[0] = (Output_local[0] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[3]) * K1_shared[3]) * K2_shared[12]));
          Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[3]) * K1_shared[3]) * K2_shared[12]));
          Output_local[2] = (Output_local[2] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[3]) * K1_shared[3]) * K2_shared[13]));
          Output_local[3] = (Output_local[3] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[3]) * K1_shared[3]) * K2_shared[13]));
          Output_local[4] = (Output_local[4] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[3]) * K1_shared[3]) * K2_shared[14]));
          Output_local[5] = (Output_local[5] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[3]) * K1_shared[3]) * K2_shared[14]));
          Output_local[6] = (Output_local[6] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[3]) * K1_shared[3]) * K2_shared[15]));
          Output_local[7] = (Output_local[7] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[3]) * K1_shared[3]) * K2_shared[15]));
          Output_local[0] = (Output_local[0] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[4]) * K1_shared[4]) * K2_shared[16]));
          Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[4]) * K1_shared[4]) * K2_shared[16]));
          Output_local[2] = (Output_local[2] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[4]) * K1_shared[4]) * K2_shared[17]));
          Output_local[3] = (Output_local[3] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[4]) * K1_shared[4]) * K2_shared[17]));
          Output_local[4] = (Output_local[4] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[4]) * K1_shared[4]) * K2_shared[18]));
          Output_local[5] = (Output_local[5] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[4]) * K1_shared[4]) * K2_shared[18]));
          Output_local[6] = (Output_local[6] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[4]) * K1_shared[4]) * K2_shared[19]));
          Output_local[7] = (Output_local[7] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[4]) * K1_shared[4]) * K2_shared[19]));
          Output_local[0] = (Output_local[0] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[5]) * K1_shared[5]) * K2_shared[20]));
          Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[5]) * K1_shared[5]) * K2_shared[20]));
          Output_local[2] = (Output_local[2] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[5]) * K1_shared[5]) * K2_shared[21]));
          Output_local[3] = (Output_local[3] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[5]) * K1_shared[5]) * K2_shared[21]));
          Output_local[4] = (Output_local[4] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[5]) * K1_shared[5]) * K2_shared[22]));
          Output_local[5] = (Output_local[5] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[5]) * K1_shared[5]) * K2_shared[22]));
          Output_local[6] = (Output_local[6] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[5]) * K1_shared[5]) * K2_shared[23]));
          Output_local[7] = (Output_local[7] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[5]) * K1_shared[5]) * K2_shared[23]));
          __syncthreads();
          pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 2))] = ((bool)1 ? Data[(((((((((nn_inner_outer * 32768) + ((((((int)threadIdx.y) * 16) + ((int)threadIdx.x)) / 64) * 16384)) + (((int)threadIdx.z) * 16384)) + (rc_outer * 1024)) + (((int)blockIdx.y) * 512)) + (hh_inner_outer * 256)) + (((((((int)threadIdx.y) * 32) + (((int)threadIdx.x) * 2)) % 128) / 16) * 32)) + (ww_inner_outer * 16)) + ((((int)threadIdx.x) * 2) % 16))] : 0.000000e+00f);
          pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 2)) + 1)] = ((bool)1 ? Data[(((((((((nn_inner_outer * 32768) + (((((((int)threadIdx.y) * 32) + (((int)threadIdx.x) * 2)) + 1) / 128) * 16384)) + (((int)threadIdx.z) * 16384)) + (rc_outer * 1024)) + (((int)blockIdx.y) * 512)) + (hh_inner_outer * 256)) + ((((((((int)threadIdx.y) * 32) + (((int)threadIdx.x) * 2)) + 1) % 128) / 16) * 32)) + (ww_inner_outer * 16)) + (((((int)threadIdx.x) * 2) + 1) % 16))] : 0.000000e+00f);
          if ((((int)threadIdx.z) * 3) < ((6 - ((int)threadIdx.y)) - ((int)threadIdx.x))) {
            if (((int)threadIdx.x) < (3 - ((int)threadIdx.y))) {
              if (((int)threadIdx.x) < 1) {
                if (((((((int)threadIdx.z) * 3) + ((int)threadIdx.x)) + ((int)threadIdx.y)) / 6) < (16 - rc_outer)) {
                  K0_shared[(((((int)threadIdx.z) * 3) + ((int)threadIdx.x)) + ((int)threadIdx.y))] = K0[((((rc_outer * 6) + (((int)threadIdx.z) * 3)) + ((int)threadIdx.x)) + ((int)threadIdx.y))];
                }
              }
            }
          }
          if ((((int)threadIdx.z) * 3) < ((6 - ((int)threadIdx.y)) - ((int)threadIdx.x))) {
            if (((int)threadIdx.x) < (3 - ((int)threadIdx.y))) {
              if (((int)threadIdx.x) < 1) {
                if ((((int)threadIdx.z) * 3) < ((12 - ((int)threadIdx.y)) - ((int)threadIdx.x))) {
                  K1_shared[(((((int)threadIdx.z) * 3) + ((int)threadIdx.x)) + ((int)threadIdx.y))] = K1[(((((((((int)threadIdx.z) * 3) + ((int)threadIdx.x)) + ((int)threadIdx.y)) / 6) * 18) + ((((((int)threadIdx.z) * 3) + ((int)threadIdx.x)) + ((int)threadIdx.y)) % 6)) + 24)];
                }
              }
            }
          }
          if ((((int)threadIdx.z) * 3) < (6 - (((((int)threadIdx.y) * 3) + ((int)threadIdx.x)) / 4))) {
            if (((((int)threadIdx.z) * 12) + (((int)threadIdx.y) * 3)) < (24 - ((int)threadIdx.x))) {
              if ((((int)threadIdx.y) * 3) < (12 - ((int)threadIdx.x))) {
                if (((int)threadIdx.x) < 3) {
                  K2_shared[(((((int)threadIdx.z) * 12) + (((int)threadIdx.y) * 3)) + ((int)threadIdx.x))] = K2[((((((int)threadIdx.z) * 48) + ((((((int)threadIdx.y) * 3) + ((int)threadIdx.x)) / 4) * 16)) + (((int)blockIdx.z) * 4)) + (((((int)threadIdx.y) * 3) + ((int)threadIdx.x)) % 4))];
                }
              }
            }
          }
          __syncthreads();
          Output_local[0] = (Output_local[0] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[0]) * K1_shared[0]) * K2_shared[0]));
          Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[0]) * K1_shared[0]) * K2_shared[0]));
          Output_local[2] = (Output_local[2] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[0]) * K1_shared[0]) * K2_shared[1]));
          Output_local[3] = (Output_local[3] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[0]) * K1_shared[0]) * K2_shared[1]));
          Output_local[4] = (Output_local[4] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[0]) * K1_shared[0]) * K2_shared[2]));
          Output_local[5] = (Output_local[5] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[0]) * K1_shared[0]) * K2_shared[2]));
          Output_local[6] = (Output_local[6] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[0]) * K1_shared[0]) * K2_shared[3]));
          Output_local[7] = (Output_local[7] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[0]) * K1_shared[0]) * K2_shared[3]));
          Output_local[0] = (Output_local[0] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[1]) * K1_shared[1]) * K2_shared[4]));
          Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[1]) * K1_shared[1]) * K2_shared[4]));
          Output_local[2] = (Output_local[2] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[1]) * K1_shared[1]) * K2_shared[5]));
          Output_local[3] = (Output_local[3] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[1]) * K1_shared[1]) * K2_shared[5]));
          Output_local[4] = (Output_local[4] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[1]) * K1_shared[1]) * K2_shared[6]));
          Output_local[5] = (Output_local[5] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[1]) * K1_shared[1]) * K2_shared[6]));
          Output_local[6] = (Output_local[6] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[1]) * K1_shared[1]) * K2_shared[7]));
          Output_local[7] = (Output_local[7] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[1]) * K1_shared[1]) * K2_shared[7]));
          Output_local[0] = (Output_local[0] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[2]) * K1_shared[2]) * K2_shared[8]));
          Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[2]) * K1_shared[2]) * K2_shared[8]));
          Output_local[2] = (Output_local[2] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[2]) * K1_shared[2]) * K2_shared[9]));
          Output_local[3] = (Output_local[3] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[2]) * K1_shared[2]) * K2_shared[9]));
          Output_local[4] = (Output_local[4] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[2]) * K1_shared[2]) * K2_shared[10]));
          Output_local[5] = (Output_local[5] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[2]) * K1_shared[2]) * K2_shared[10]));
          Output_local[6] = (Output_local[6] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[2]) * K1_shared[2]) * K2_shared[11]));
          Output_local[7] = (Output_local[7] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[2]) * K1_shared[2]) * K2_shared[11]));
          Output_local[0] = (Output_local[0] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[3]) * K1_shared[3]) * K2_shared[12]));
          Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[3]) * K1_shared[3]) * K2_shared[12]));
          Output_local[2] = (Output_local[2] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[3]) * K1_shared[3]) * K2_shared[13]));
          Output_local[3] = (Output_local[3] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[3]) * K1_shared[3]) * K2_shared[13]));
          Output_local[4] = (Output_local[4] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[3]) * K1_shared[3]) * K2_shared[14]));
          Output_local[5] = (Output_local[5] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[3]) * K1_shared[3]) * K2_shared[14]));
          Output_local[6] = (Output_local[6] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[3]) * K1_shared[3]) * K2_shared[15]));
          Output_local[7] = (Output_local[7] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[3]) * K1_shared[3]) * K2_shared[15]));
          Output_local[0] = (Output_local[0] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[4]) * K1_shared[4]) * K2_shared[16]));
          Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[4]) * K1_shared[4]) * K2_shared[16]));
          Output_local[2] = (Output_local[2] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[4]) * K1_shared[4]) * K2_shared[17]));
          Output_local[3] = (Output_local[3] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[4]) * K1_shared[4]) * K2_shared[17]));
          Output_local[4] = (Output_local[4] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[4]) * K1_shared[4]) * K2_shared[18]));
          Output_local[5] = (Output_local[5] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[4]) * K1_shared[4]) * K2_shared[18]));
          Output_local[6] = (Output_local[6] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[4]) * K1_shared[4]) * K2_shared[19]));
          Output_local[7] = (Output_local[7] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[4]) * K1_shared[4]) * K2_shared[19]));
          Output_local[0] = (Output_local[0] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[5]) * K1_shared[5]) * K2_shared[20]));
          Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[5]) * K1_shared[5]) * K2_shared[20]));
          Output_local[2] = (Output_local[2] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[5]) * K1_shared[5]) * K2_shared[21]));
          Output_local[3] = (Output_local[3] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[5]) * K1_shared[5]) * K2_shared[21]));
          Output_local[4] = (Output_local[4] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[5]) * K1_shared[5]) * K2_shared[22]));
          Output_local[5] = (Output_local[5] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[5]) * K1_shared[5]) * K2_shared[22]));
          Output_local[6] = (Output_local[6] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[5]) * K1_shared[5]) * K2_shared[23]));
          Output_local[7] = (Output_local[7] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[5]) * K1_shared[5]) * K2_shared[23]));
          __syncthreads();
          pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 2))] = (((ww_inner_outer * 16) < (31 - ((((int)threadIdx.x) * 2) % 16))) ? Data[((((((((((nn_inner_outer * 32768) + ((((((int)threadIdx.y) * 16) + ((int)threadIdx.x)) / 64) * 16384)) + (((int)threadIdx.z) * 16384)) + (rc_outer * 1024)) + (((int)blockIdx.y) * 512)) + (hh_inner_outer * 256)) + (((((((int)threadIdx.y) * 32) + (((int)threadIdx.x) * 2)) % 128) / 16) * 32)) + (ww_inner_outer * 16)) + ((((int)threadIdx.x) * 2) % 16)) + 1)] : 0.000000e+00f);
          pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 2)) + 1)] = (((ww_inner_outer * 16) < (31 - (((((int)threadIdx.x) * 2) + 1) % 16))) ? Data[((((((((((nn_inner_outer * 32768) + (((((((int)threadIdx.y) * 32) + (((int)threadIdx.x) * 2)) + 1) / 128) * 16384)) + (((int)threadIdx.z) * 16384)) + (rc_outer * 1024)) + (((int)blockIdx.y) * 512)) + (hh_inner_outer * 256)) + ((((((((int)threadIdx.y) * 32) + (((int)threadIdx.x) * 2)) + 1) % 128) / 16) * 32)) + (ww_inner_outer * 16)) + (((((int)threadIdx.x) * 2) + 1) % 16)) + 1)] : 0.000000e+00f);
          if ((((int)threadIdx.z) * 3) < ((6 - ((int)threadIdx.y)) - ((int)threadIdx.x))) {
            if (((int)threadIdx.x) < (3 - ((int)threadIdx.y))) {
              if (((int)threadIdx.x) < 1) {
                if (((((((int)threadIdx.z) * 3) + ((int)threadIdx.x)) + ((int)threadIdx.y)) / 6) < (16 - rc_outer)) {
                  K0_shared[(((((int)threadIdx.z) * 3) + ((int)threadIdx.x)) + ((int)threadIdx.y))] = K0[((((rc_outer * 6) + (((int)threadIdx.z) * 3)) + ((int)threadIdx.x)) + ((int)threadIdx.y))];
                }
              }
            }
          }
          if ((((int)threadIdx.z) * 3) < ((6 - ((int)threadIdx.y)) - ((int)threadIdx.x))) {
            if (((int)threadIdx.x) < (3 - ((int)threadIdx.y))) {
              if (((int)threadIdx.x) < 1) {
                if ((((int)threadIdx.z) * 3) < ((12 - ((int)threadIdx.y)) - ((int)threadIdx.x))) {
                  K1_shared[(((((int)threadIdx.z) * 3) + ((int)threadIdx.x)) + ((int)threadIdx.y))] = K1[(((((((((int)threadIdx.z) * 3) + ((int)threadIdx.x)) + ((int)threadIdx.y)) / 6) * 18) + ((((((int)threadIdx.z) * 3) + ((int)threadIdx.x)) + ((int)threadIdx.y)) % 6)) + 30)];
                }
              }
            }
          }
          if ((((int)threadIdx.z) * 3) < (6 - (((((int)threadIdx.y) * 3) + ((int)threadIdx.x)) / 4))) {
            if (((((int)threadIdx.z) * 12) + (((int)threadIdx.y) * 3)) < (24 - ((int)threadIdx.x))) {
              if ((((int)threadIdx.y) * 3) < (12 - ((int)threadIdx.x))) {
                if (((int)threadIdx.x) < 3) {
                  K2_shared[(((((int)threadIdx.z) * 12) + (((int)threadIdx.y) * 3)) + ((int)threadIdx.x))] = K2[((((((int)threadIdx.z) * 48) + ((((((int)threadIdx.y) * 3) + ((int)threadIdx.x)) / 4) * 16)) + (((int)blockIdx.z) * 4)) + (((((int)threadIdx.y) * 3) + ((int)threadIdx.x)) % 4))];
                }
              }
            }
          }
          __syncthreads();
          Output_local[0] = (Output_local[0] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[0]) * K1_shared[0]) * K2_shared[0]));
          Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[0]) * K1_shared[0]) * K2_shared[0]));
          Output_local[2] = (Output_local[2] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[0]) * K1_shared[0]) * K2_shared[1]));
          Output_local[3] = (Output_local[3] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[0]) * K1_shared[0]) * K2_shared[1]));
          Output_local[4] = (Output_local[4] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[0]) * K1_shared[0]) * K2_shared[2]));
          Output_local[5] = (Output_local[5] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[0]) * K1_shared[0]) * K2_shared[2]));
          Output_local[6] = (Output_local[6] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[0]) * K1_shared[0]) * K2_shared[3]));
          Output_local[7] = (Output_local[7] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[0]) * K1_shared[0]) * K2_shared[3]));
          Output_local[0] = (Output_local[0] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[1]) * K1_shared[1]) * K2_shared[4]));
          Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[1]) * K1_shared[1]) * K2_shared[4]));
          Output_local[2] = (Output_local[2] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[1]) * K1_shared[1]) * K2_shared[5]));
          Output_local[3] = (Output_local[3] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[1]) * K1_shared[1]) * K2_shared[5]));
          Output_local[4] = (Output_local[4] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[1]) * K1_shared[1]) * K2_shared[6]));
          Output_local[5] = (Output_local[5] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[1]) * K1_shared[1]) * K2_shared[6]));
          Output_local[6] = (Output_local[6] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[1]) * K1_shared[1]) * K2_shared[7]));
          Output_local[7] = (Output_local[7] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[1]) * K1_shared[1]) * K2_shared[7]));
          Output_local[0] = (Output_local[0] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[2]) * K1_shared[2]) * K2_shared[8]));
          Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[2]) * K1_shared[2]) * K2_shared[8]));
          Output_local[2] = (Output_local[2] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[2]) * K1_shared[2]) * K2_shared[9]));
          Output_local[3] = (Output_local[3] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[2]) * K1_shared[2]) * K2_shared[9]));
          Output_local[4] = (Output_local[4] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[2]) * K1_shared[2]) * K2_shared[10]));
          Output_local[5] = (Output_local[5] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[2]) * K1_shared[2]) * K2_shared[10]));
          Output_local[6] = (Output_local[6] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[2]) * K1_shared[2]) * K2_shared[11]));
          Output_local[7] = (Output_local[7] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[2]) * K1_shared[2]) * K2_shared[11]));
          Output_local[0] = (Output_local[0] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[3]) * K1_shared[3]) * K2_shared[12]));
          Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[3]) * K1_shared[3]) * K2_shared[12]));
          Output_local[2] = (Output_local[2] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[3]) * K1_shared[3]) * K2_shared[13]));
          Output_local[3] = (Output_local[3] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[3]) * K1_shared[3]) * K2_shared[13]));
          Output_local[4] = (Output_local[4] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[3]) * K1_shared[3]) * K2_shared[14]));
          Output_local[5] = (Output_local[5] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[3]) * K1_shared[3]) * K2_shared[14]));
          Output_local[6] = (Output_local[6] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[3]) * K1_shared[3]) * K2_shared[15]));
          Output_local[7] = (Output_local[7] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[3]) * K1_shared[3]) * K2_shared[15]));
          Output_local[0] = (Output_local[0] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[4]) * K1_shared[4]) * K2_shared[16]));
          Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[4]) * K1_shared[4]) * K2_shared[16]));
          Output_local[2] = (Output_local[2] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[4]) * K1_shared[4]) * K2_shared[17]));
          Output_local[3] = (Output_local[3] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[4]) * K1_shared[4]) * K2_shared[17]));
          Output_local[4] = (Output_local[4] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[4]) * K1_shared[4]) * K2_shared[18]));
          Output_local[5] = (Output_local[5] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[4]) * K1_shared[4]) * K2_shared[18]));
          Output_local[6] = (Output_local[6] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[4]) * K1_shared[4]) * K2_shared[19]));
          Output_local[7] = (Output_local[7] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[4]) * K1_shared[4]) * K2_shared[19]));
          Output_local[0] = (Output_local[0] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[5]) * K1_shared[5]) * K2_shared[20]));
          Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[5]) * K1_shared[5]) * K2_shared[20]));
          Output_local[2] = (Output_local[2] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[5]) * K1_shared[5]) * K2_shared[21]));
          Output_local[3] = (Output_local[3] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[5]) * K1_shared[5]) * K2_shared[21]));
          Output_local[4] = (Output_local[4] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[5]) * K1_shared[5]) * K2_shared[22]));
          Output_local[5] = (Output_local[5] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[5]) * K1_shared[5]) * K2_shared[22]));
          Output_local[6] = (Output_local[6] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[5]) * K1_shared[5]) * K2_shared[23]));
          Output_local[7] = (Output_local[7] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[5]) * K1_shared[5]) * K2_shared[23]));
          __syncthreads();
          pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 2))] = (((((((int)blockIdx.y) * 16) + (hh_inner_outer * 8)) < ((31 - (((int)threadIdx.x) / 8)) - (((int)threadIdx.y) * 2))) && ((1 - ((((int)threadIdx.x) * 2) % 16)) <= (ww_inner_outer * 16))) ? Data[((((((((((nn_inner_outer * 32768) + ((((((int)threadIdx.y) * 16) + ((int)threadIdx.x)) / 64) * 16384)) + (((int)threadIdx.z) * 16384)) + (rc_outer * 1024)) + (((int)blockIdx.y) * 512)) + (hh_inner_outer * 256)) + (((((((int)threadIdx.y) * 32) + (((int)threadIdx.x) * 2)) % 128) / 16) * 32)) + (ww_inner_outer * 16)) + ((((int)threadIdx.x) * 2) % 16)) + 31)] : 0.000000e+00f);
          pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 2)) + 1)] = (((((((int)blockIdx.y) * 16) + (hh_inner_outer * 8)) < ((31 - (((((int)threadIdx.x) * 2) + 1) / 16)) - (((int)threadIdx.y) * 2))) && ((1 - (((((int)threadIdx.x) * 2) + 1) % 16)) <= (ww_inner_outer * 16))) ? Data[((((((((((nn_inner_outer * 32768) + (((((((int)threadIdx.y) * 32) + (((int)threadIdx.x) * 2)) + 1) / 128) * 16384)) + (((int)threadIdx.z) * 16384)) + (rc_outer * 1024)) + (((int)blockIdx.y) * 512)) + (hh_inner_outer * 256)) + ((((((((int)threadIdx.y) * 32) + (((int)threadIdx.x) * 2)) + 1) % 128) / 16) * 32)) + (ww_inner_outer * 16)) + (((((int)threadIdx.x) * 2) + 1) % 16)) + 31)] : 0.000000e+00f);
          if ((((int)threadIdx.z) * 3) < ((6 - ((int)threadIdx.y)) - ((int)threadIdx.x))) {
            if (((int)threadIdx.x) < (3 - ((int)threadIdx.y))) {
              if (((int)threadIdx.x) < 1) {
                if (((((((int)threadIdx.z) * 3) + ((int)threadIdx.x)) + ((int)threadIdx.y)) / 6) < (16 - rc_outer)) {
                  K0_shared[(((((int)threadIdx.z) * 3) + ((int)threadIdx.x)) + ((int)threadIdx.y))] = K0[((((rc_outer * 6) + (((int)threadIdx.z) * 3)) + ((int)threadIdx.x)) + ((int)threadIdx.y))];
                }
              }
            }
          }
          if ((((int)threadIdx.z) * 3) < ((6 - ((int)threadIdx.y)) - ((int)threadIdx.x))) {
            if (((int)threadIdx.x) < (3 - ((int)threadIdx.y))) {
              if (((int)threadIdx.x) < 1) {
                K1_shared[(((((int)threadIdx.z) * 3) + ((int)threadIdx.x)) + ((int)threadIdx.y))] = K1[(((((((((int)threadIdx.z) * 3) + ((int)threadIdx.x)) + ((int)threadIdx.y)) / 6) * 18) + ((((((int)threadIdx.z) * 3) + ((int)threadIdx.x)) + ((int)threadIdx.y)) % 6)) + 36)];
              }
            }
          }
          if ((((int)threadIdx.z) * 3) < (6 - (((((int)threadIdx.y) * 3) + ((int)threadIdx.x)) / 4))) {
            if (((((int)threadIdx.z) * 12) + (((int)threadIdx.y) * 3)) < (24 - ((int)threadIdx.x))) {
              if ((((int)threadIdx.y) * 3) < (12 - ((int)threadIdx.x))) {
                if (((int)threadIdx.x) < 3) {
                  K2_shared[(((((int)threadIdx.z) * 12) + (((int)threadIdx.y) * 3)) + ((int)threadIdx.x))] = K2[((((((int)threadIdx.z) * 48) + ((((((int)threadIdx.y) * 3) + ((int)threadIdx.x)) / 4) * 16)) + (((int)blockIdx.z) * 4)) + (((((int)threadIdx.y) * 3) + ((int)threadIdx.x)) % 4))];
                }
              }
            }
          }
          __syncthreads();
          Output_local[0] = (Output_local[0] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[0]) * K1_shared[0]) * K2_shared[0]));
          Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[0]) * K1_shared[0]) * K2_shared[0]));
          Output_local[2] = (Output_local[2] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[0]) * K1_shared[0]) * K2_shared[1]));
          Output_local[3] = (Output_local[3] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[0]) * K1_shared[0]) * K2_shared[1]));
          Output_local[4] = (Output_local[4] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[0]) * K1_shared[0]) * K2_shared[2]));
          Output_local[5] = (Output_local[5] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[0]) * K1_shared[0]) * K2_shared[2]));
          Output_local[6] = (Output_local[6] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[0]) * K1_shared[0]) * K2_shared[3]));
          Output_local[7] = (Output_local[7] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[0]) * K1_shared[0]) * K2_shared[3]));
          Output_local[0] = (Output_local[0] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[1]) * K1_shared[1]) * K2_shared[4]));
          Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[1]) * K1_shared[1]) * K2_shared[4]));
          Output_local[2] = (Output_local[2] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[1]) * K1_shared[1]) * K2_shared[5]));
          Output_local[3] = (Output_local[3] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[1]) * K1_shared[1]) * K2_shared[5]));
          Output_local[4] = (Output_local[4] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[1]) * K1_shared[1]) * K2_shared[6]));
          Output_local[5] = (Output_local[5] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[1]) * K1_shared[1]) * K2_shared[6]));
          Output_local[6] = (Output_local[6] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[1]) * K1_shared[1]) * K2_shared[7]));
          Output_local[7] = (Output_local[7] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[1]) * K1_shared[1]) * K2_shared[7]));
          Output_local[0] = (Output_local[0] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[2]) * K1_shared[2]) * K2_shared[8]));
          Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[2]) * K1_shared[2]) * K2_shared[8]));
          Output_local[2] = (Output_local[2] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[2]) * K1_shared[2]) * K2_shared[9]));
          Output_local[3] = (Output_local[3] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[2]) * K1_shared[2]) * K2_shared[9]));
          Output_local[4] = (Output_local[4] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[2]) * K1_shared[2]) * K2_shared[10]));
          Output_local[5] = (Output_local[5] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[2]) * K1_shared[2]) * K2_shared[10]));
          Output_local[6] = (Output_local[6] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[2]) * K1_shared[2]) * K2_shared[11]));
          Output_local[7] = (Output_local[7] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[2]) * K1_shared[2]) * K2_shared[11]));
          Output_local[0] = (Output_local[0] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[3]) * K1_shared[3]) * K2_shared[12]));
          Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[3]) * K1_shared[3]) * K2_shared[12]));
          Output_local[2] = (Output_local[2] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[3]) * K1_shared[3]) * K2_shared[13]));
          Output_local[3] = (Output_local[3] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[3]) * K1_shared[3]) * K2_shared[13]));
          Output_local[4] = (Output_local[4] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[3]) * K1_shared[3]) * K2_shared[14]));
          Output_local[5] = (Output_local[5] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[3]) * K1_shared[3]) * K2_shared[14]));
          Output_local[6] = (Output_local[6] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[3]) * K1_shared[3]) * K2_shared[15]));
          Output_local[7] = (Output_local[7] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[3]) * K1_shared[3]) * K2_shared[15]));
          Output_local[0] = (Output_local[0] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[4]) * K1_shared[4]) * K2_shared[16]));
          Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[4]) * K1_shared[4]) * K2_shared[16]));
          Output_local[2] = (Output_local[2] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[4]) * K1_shared[4]) * K2_shared[17]));
          Output_local[3] = (Output_local[3] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[4]) * K1_shared[4]) * K2_shared[17]));
          Output_local[4] = (Output_local[4] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[4]) * K1_shared[4]) * K2_shared[18]));
          Output_local[5] = (Output_local[5] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[4]) * K1_shared[4]) * K2_shared[18]));
          Output_local[6] = (Output_local[6] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[4]) * K1_shared[4]) * K2_shared[19]));
          Output_local[7] = (Output_local[7] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[4]) * K1_shared[4]) * K2_shared[19]));
          Output_local[0] = (Output_local[0] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[5]) * K1_shared[5]) * K2_shared[20]));
          Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[5]) * K1_shared[5]) * K2_shared[20]));
          Output_local[2] = (Output_local[2] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[5]) * K1_shared[5]) * K2_shared[21]));
          Output_local[3] = (Output_local[3] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[5]) * K1_shared[5]) * K2_shared[21]));
          Output_local[4] = (Output_local[4] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[5]) * K1_shared[5]) * K2_shared[22]));
          Output_local[5] = (Output_local[5] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[5]) * K1_shared[5]) * K2_shared[22]));
          Output_local[6] = (Output_local[6] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[5]) * K1_shared[5]) * K2_shared[23]));
          Output_local[7] = (Output_local[7] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[5]) * K1_shared[5]) * K2_shared[23]));
          __syncthreads();
          pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 2))] = ((((((int)blockIdx.y) * 16) + (hh_inner_outer * 8)) < ((31 - (((int)threadIdx.x) / 8)) - (((int)threadIdx.y) * 2))) ? Data[((((((((((nn_inner_outer * 32768) + ((((((int)threadIdx.y) * 16) + ((int)threadIdx.x)) / 64) * 16384)) + (((int)threadIdx.z) * 16384)) + (rc_outer * 1024)) + (((int)blockIdx.y) * 512)) + (hh_inner_outer * 256)) + (((((((int)threadIdx.y) * 32) + (((int)threadIdx.x) * 2)) % 128) / 16) * 32)) + (ww_inner_outer * 16)) + ((((int)threadIdx.x) * 2) % 16)) + 32)] : 0.000000e+00f);
          pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 2)) + 1)] = ((((((int)blockIdx.y) * 16) + (hh_inner_outer * 8)) < ((31 - (((((int)threadIdx.x) * 2) + 1) / 16)) - (((int)threadIdx.y) * 2))) ? Data[((((((((((nn_inner_outer * 32768) + (((((((int)threadIdx.y) * 32) + (((int)threadIdx.x) * 2)) + 1) / 128) * 16384)) + (((int)threadIdx.z) * 16384)) + (rc_outer * 1024)) + (((int)blockIdx.y) * 512)) + (hh_inner_outer * 256)) + ((((((((int)threadIdx.y) * 32) + (((int)threadIdx.x) * 2)) + 1) % 128) / 16) * 32)) + (ww_inner_outer * 16)) + (((((int)threadIdx.x) * 2) + 1) % 16)) + 32)] : 0.000000e+00f);
          if ((((int)threadIdx.z) * 3) < ((6 - ((int)threadIdx.y)) - ((int)threadIdx.x))) {
            if (((int)threadIdx.x) < (3 - ((int)threadIdx.y))) {
              if (((int)threadIdx.x) < 1) {
                if (((((((int)threadIdx.z) * 3) + ((int)threadIdx.x)) + ((int)threadIdx.y)) / 6) < (16 - rc_outer)) {
                  K0_shared[(((((int)threadIdx.z) * 3) + ((int)threadIdx.x)) + ((int)threadIdx.y))] = K0[((((rc_outer * 6) + (((int)threadIdx.z) * 3)) + ((int)threadIdx.x)) + ((int)threadIdx.y))];
                }
              }
            }
          }
          if ((((int)threadIdx.z) * 3) < ((6 - ((int)threadIdx.y)) - ((int)threadIdx.x))) {
            if (((int)threadIdx.x) < (3 - ((int)threadIdx.y))) {
              if (((int)threadIdx.x) < 1) {
                K1_shared[(((((int)threadIdx.z) * 3) + ((int)threadIdx.x)) + ((int)threadIdx.y))] = K1[(((((((((int)threadIdx.z) * 3) + ((int)threadIdx.x)) + ((int)threadIdx.y)) / 6) * 18) + ((((((int)threadIdx.z) * 3) + ((int)threadIdx.x)) + ((int)threadIdx.y)) % 6)) + 42)];
              }
            }
          }
          if ((((int)threadIdx.z) * 3) < (6 - (((((int)threadIdx.y) * 3) + ((int)threadIdx.x)) / 4))) {
            if (((((int)threadIdx.z) * 12) + (((int)threadIdx.y) * 3)) < (24 - ((int)threadIdx.x))) {
              if ((((int)threadIdx.y) * 3) < (12 - ((int)threadIdx.x))) {
                if (((int)threadIdx.x) < 3) {
                  K2_shared[(((((int)threadIdx.z) * 12) + (((int)threadIdx.y) * 3)) + ((int)threadIdx.x))] = K2[((((((int)threadIdx.z) * 48) + ((((((int)threadIdx.y) * 3) + ((int)threadIdx.x)) / 4) * 16)) + (((int)blockIdx.z) * 4)) + (((((int)threadIdx.y) * 3) + ((int)threadIdx.x)) % 4))];
                }
              }
            }
          }
          __syncthreads();
          Output_local[0] = (Output_local[0] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[0]) * K1_shared[0]) * K2_shared[0]));
          Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[0]) * K1_shared[0]) * K2_shared[0]));
          Output_local[2] = (Output_local[2] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[0]) * K1_shared[0]) * K2_shared[1]));
          Output_local[3] = (Output_local[3] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[0]) * K1_shared[0]) * K2_shared[1]));
          Output_local[4] = (Output_local[4] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[0]) * K1_shared[0]) * K2_shared[2]));
          Output_local[5] = (Output_local[5] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[0]) * K1_shared[0]) * K2_shared[2]));
          Output_local[6] = (Output_local[6] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[0]) * K1_shared[0]) * K2_shared[3]));
          Output_local[7] = (Output_local[7] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[0]) * K1_shared[0]) * K2_shared[3]));
          Output_local[0] = (Output_local[0] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[1]) * K1_shared[1]) * K2_shared[4]));
          Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[1]) * K1_shared[1]) * K2_shared[4]));
          Output_local[2] = (Output_local[2] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[1]) * K1_shared[1]) * K2_shared[5]));
          Output_local[3] = (Output_local[3] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[1]) * K1_shared[1]) * K2_shared[5]));
          Output_local[4] = (Output_local[4] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[1]) * K1_shared[1]) * K2_shared[6]));
          Output_local[5] = (Output_local[5] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[1]) * K1_shared[1]) * K2_shared[6]));
          Output_local[6] = (Output_local[6] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[1]) * K1_shared[1]) * K2_shared[7]));
          Output_local[7] = (Output_local[7] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[1]) * K1_shared[1]) * K2_shared[7]));
          Output_local[0] = (Output_local[0] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[2]) * K1_shared[2]) * K2_shared[8]));
          Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[2]) * K1_shared[2]) * K2_shared[8]));
          Output_local[2] = (Output_local[2] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[2]) * K1_shared[2]) * K2_shared[9]));
          Output_local[3] = (Output_local[3] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[2]) * K1_shared[2]) * K2_shared[9]));
          Output_local[4] = (Output_local[4] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[2]) * K1_shared[2]) * K2_shared[10]));
          Output_local[5] = (Output_local[5] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[2]) * K1_shared[2]) * K2_shared[10]));
          Output_local[6] = (Output_local[6] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[2]) * K1_shared[2]) * K2_shared[11]));
          Output_local[7] = (Output_local[7] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[2]) * K1_shared[2]) * K2_shared[11]));
          Output_local[0] = (Output_local[0] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[3]) * K1_shared[3]) * K2_shared[12]));
          Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[3]) * K1_shared[3]) * K2_shared[12]));
          Output_local[2] = (Output_local[2] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[3]) * K1_shared[3]) * K2_shared[13]));
          Output_local[3] = (Output_local[3] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[3]) * K1_shared[3]) * K2_shared[13]));
          Output_local[4] = (Output_local[4] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[3]) * K1_shared[3]) * K2_shared[14]));
          Output_local[5] = (Output_local[5] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[3]) * K1_shared[3]) * K2_shared[14]));
          Output_local[6] = (Output_local[6] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[3]) * K1_shared[3]) * K2_shared[15]));
          Output_local[7] = (Output_local[7] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[3]) * K1_shared[3]) * K2_shared[15]));
          Output_local[0] = (Output_local[0] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[4]) * K1_shared[4]) * K2_shared[16]));
          Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[4]) * K1_shared[4]) * K2_shared[16]));
          Output_local[2] = (Output_local[2] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[4]) * K1_shared[4]) * K2_shared[17]));
          Output_local[3] = (Output_local[3] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[4]) * K1_shared[4]) * K2_shared[17]));
          Output_local[4] = (Output_local[4] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[4]) * K1_shared[4]) * K2_shared[18]));
          Output_local[5] = (Output_local[5] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[4]) * K1_shared[4]) * K2_shared[18]));
          Output_local[6] = (Output_local[6] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[4]) * K1_shared[4]) * K2_shared[19]));
          Output_local[7] = (Output_local[7] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[4]) * K1_shared[4]) * K2_shared[19]));
          Output_local[0] = (Output_local[0] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[5]) * K1_shared[5]) * K2_shared[20]));
          Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[5]) * K1_shared[5]) * K2_shared[20]));
          Output_local[2] = (Output_local[2] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[5]) * K1_shared[5]) * K2_shared[21]));
          Output_local[3] = (Output_local[3] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[5]) * K1_shared[5]) * K2_shared[21]));
          Output_local[4] = (Output_local[4] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[5]) * K1_shared[5]) * K2_shared[22]));
          Output_local[5] = (Output_local[5] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[5]) * K1_shared[5]) * K2_shared[22]));
          Output_local[6] = (Output_local[6] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[5]) * K1_shared[5]) * K2_shared[23]));
          Output_local[7] = (Output_local[7] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[5]) * K1_shared[5]) * K2_shared[23]));
          __syncthreads();
          pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 2))] = (((((((int)blockIdx.y) * 16) + (hh_inner_outer * 8)) < ((31 - (((int)threadIdx.x) / 8)) - (((int)threadIdx.y) * 2))) && ((ww_inner_outer * 16) < (31 - ((((int)threadIdx.x) * 2) % 16)))) ? Data[((((((((((nn_inner_outer * 32768) + ((((((int)threadIdx.y) * 16) + ((int)threadIdx.x)) / 64) * 16384)) + (((int)threadIdx.z) * 16384)) + (rc_outer * 1024)) + (((int)blockIdx.y) * 512)) + (hh_inner_outer * 256)) + (((((((int)threadIdx.y) * 32) + (((int)threadIdx.x) * 2)) % 128) / 16) * 32)) + (ww_inner_outer * 16)) + ((((int)threadIdx.x) * 2) % 16)) + 33)] : 0.000000e+00f);
          pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 2)) + 1)] = (((((((int)blockIdx.y) * 16) + (hh_inner_outer * 8)) < ((31 - (((((int)threadIdx.x) * 2) + 1) / 16)) - (((int)threadIdx.y) * 2))) && ((ww_inner_outer * 16) < (31 - (((((int)threadIdx.x) * 2) + 1) % 16)))) ? Data[((((((((((nn_inner_outer * 32768) + (((((((int)threadIdx.y) * 32) + (((int)threadIdx.x) * 2)) + 1) / 128) * 16384)) + (((int)threadIdx.z) * 16384)) + (rc_outer * 1024)) + (((int)blockIdx.y) * 512)) + (hh_inner_outer * 256)) + ((((((((int)threadIdx.y) * 32) + (((int)threadIdx.x) * 2)) + 1) % 128) / 16) * 32)) + (ww_inner_outer * 16)) + (((((int)threadIdx.x) * 2) + 1) % 16)) + 33)] : 0.000000e+00f);
          if ((((int)threadIdx.z) * 3) < ((6 - ((int)threadIdx.y)) - ((int)threadIdx.x))) {
            if (((int)threadIdx.x) < (3 - ((int)threadIdx.y))) {
              if (((int)threadIdx.x) < 1) {
                if (((((((int)threadIdx.z) * 3) + ((int)threadIdx.x)) + ((int)threadIdx.y)) / 6) < (16 - rc_outer)) {
                  K0_shared[(((((int)threadIdx.z) * 3) + ((int)threadIdx.x)) + ((int)threadIdx.y))] = K0[((((rc_outer * 6) + (((int)threadIdx.z) * 3)) + ((int)threadIdx.x)) + ((int)threadIdx.y))];
                }
              }
            }
          }
          if ((((int)threadIdx.z) * 3) < ((6 - ((int)threadIdx.y)) - ((int)threadIdx.x))) {
            if (((int)threadIdx.x) < (3 - ((int)threadIdx.y))) {
              if (((int)threadIdx.x) < 1) {
                K1_shared[(((((int)threadIdx.z) * 3) + ((int)threadIdx.x)) + ((int)threadIdx.y))] = K1[(((((((((int)threadIdx.z) * 3) + ((int)threadIdx.x)) + ((int)threadIdx.y)) / 6) * 18) + ((((((int)threadIdx.z) * 3) + ((int)threadIdx.x)) + ((int)threadIdx.y)) % 6)) + 48)];
              }
            }
          }
          if ((((int)threadIdx.z) * 3) < (6 - (((((int)threadIdx.y) * 3) + ((int)threadIdx.x)) / 4))) {
            if (((((int)threadIdx.z) * 12) + (((int)threadIdx.y) * 3)) < (24 - ((int)threadIdx.x))) {
              if ((((int)threadIdx.y) * 3) < (12 - ((int)threadIdx.x))) {
                if (((int)threadIdx.x) < 3) {
                  K2_shared[(((((int)threadIdx.z) * 12) + (((int)threadIdx.y) * 3)) + ((int)threadIdx.x))] = K2[((((((int)threadIdx.z) * 48) + ((((((int)threadIdx.y) * 3) + ((int)threadIdx.x)) / 4) * 16)) + (((int)blockIdx.z) * 4)) + (((((int)threadIdx.y) * 3) + ((int)threadIdx.x)) % 4))];
                }
              }
            }
          }
          __syncthreads();
          Output_local[0] = (Output_local[0] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[0]) * K1_shared[0]) * K2_shared[0]));
          Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[0]) * K1_shared[0]) * K2_shared[0]));
          Output_local[2] = (Output_local[2] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[0]) * K1_shared[0]) * K2_shared[1]));
          Output_local[3] = (Output_local[3] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[0]) * K1_shared[0]) * K2_shared[1]));
          Output_local[4] = (Output_local[4] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[0]) * K1_shared[0]) * K2_shared[2]));
          Output_local[5] = (Output_local[5] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[0]) * K1_shared[0]) * K2_shared[2]));
          Output_local[6] = (Output_local[6] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[0]) * K1_shared[0]) * K2_shared[3]));
          Output_local[7] = (Output_local[7] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[0]) * K1_shared[0]) * K2_shared[3]));
          Output_local[0] = (Output_local[0] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[1]) * K1_shared[1]) * K2_shared[4]));
          Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[1]) * K1_shared[1]) * K2_shared[4]));
          Output_local[2] = (Output_local[2] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[1]) * K1_shared[1]) * K2_shared[5]));
          Output_local[3] = (Output_local[3] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[1]) * K1_shared[1]) * K2_shared[5]));
          Output_local[4] = (Output_local[4] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[1]) * K1_shared[1]) * K2_shared[6]));
          Output_local[5] = (Output_local[5] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[1]) * K1_shared[1]) * K2_shared[6]));
          Output_local[6] = (Output_local[6] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[1]) * K1_shared[1]) * K2_shared[7]));
          Output_local[7] = (Output_local[7] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[1]) * K1_shared[1]) * K2_shared[7]));
          Output_local[0] = (Output_local[0] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[2]) * K1_shared[2]) * K2_shared[8]));
          Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[2]) * K1_shared[2]) * K2_shared[8]));
          Output_local[2] = (Output_local[2] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[2]) * K1_shared[2]) * K2_shared[9]));
          Output_local[3] = (Output_local[3] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[2]) * K1_shared[2]) * K2_shared[9]));
          Output_local[4] = (Output_local[4] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[2]) * K1_shared[2]) * K2_shared[10]));
          Output_local[5] = (Output_local[5] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[2]) * K1_shared[2]) * K2_shared[10]));
          Output_local[6] = (Output_local[6] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[2]) * K1_shared[2]) * K2_shared[11]));
          Output_local[7] = (Output_local[7] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[2]) * K1_shared[2]) * K2_shared[11]));
          Output_local[0] = (Output_local[0] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[3]) * K1_shared[3]) * K2_shared[12]));
          Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[3]) * K1_shared[3]) * K2_shared[12]));
          Output_local[2] = (Output_local[2] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[3]) * K1_shared[3]) * K2_shared[13]));
          Output_local[3] = (Output_local[3] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[3]) * K1_shared[3]) * K2_shared[13]));
          Output_local[4] = (Output_local[4] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[3]) * K1_shared[3]) * K2_shared[14]));
          Output_local[5] = (Output_local[5] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[3]) * K1_shared[3]) * K2_shared[14]));
          Output_local[6] = (Output_local[6] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[3]) * K1_shared[3]) * K2_shared[15]));
          Output_local[7] = (Output_local[7] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[3]) * K1_shared[3]) * K2_shared[15]));
          Output_local[0] = (Output_local[0] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[4]) * K1_shared[4]) * K2_shared[16]));
          Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[4]) * K1_shared[4]) * K2_shared[16]));
          Output_local[2] = (Output_local[2] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[4]) * K1_shared[4]) * K2_shared[17]));
          Output_local[3] = (Output_local[3] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[4]) * K1_shared[4]) * K2_shared[17]));
          Output_local[4] = (Output_local[4] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[4]) * K1_shared[4]) * K2_shared[18]));
          Output_local[5] = (Output_local[5] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[4]) * K1_shared[4]) * K2_shared[18]));
          Output_local[6] = (Output_local[6] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[4]) * K1_shared[4]) * K2_shared[19]));
          Output_local[7] = (Output_local[7] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[4]) * K1_shared[4]) * K2_shared[19]));
          Output_local[0] = (Output_local[0] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[5]) * K1_shared[5]) * K2_shared[20]));
          Output_local[1] = (Output_local[1] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[5]) * K1_shared[5]) * K2_shared[20]));
          Output_local[2] = (Output_local[2] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[5]) * K1_shared[5]) * K2_shared[21]));
          Output_local[3] = (Output_local[3] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[5]) * K1_shared[5]) * K2_shared[21]));
          Output_local[4] = (Output_local[4] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[5]) * K1_shared[5]) * K2_shared[22]));
          Output_local[5] = (Output_local[5] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[5]) * K1_shared[5]) * K2_shared[22]));
          Output_local[6] = (Output_local[6] + (((pad_temp_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x))] * K0_shared[5]) * K1_shared[5]) * K2_shared[23]));
          Output_local[7] = (Output_local[7] + (((pad_temp_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) + 16)] * K0_shared[5]) * K1_shared[5]) * K2_shared[23]));
        }
        Output[((((((((nn_inner_outer * 32768) + (((int)threadIdx.z) * 16384)) + (((int)blockIdx.z) * 4096)) + (((int)blockIdx.y) * 512)) + (hh_inner_outer * 256)) + (((int)threadIdx.y) * 64)) + (ww_inner_outer * 16)) + ((int)threadIdx.x))] = Output_local[0];
        Output[(((((((((nn_inner_outer * 32768) + (((int)threadIdx.z) * 16384)) + (((int)blockIdx.z) * 4096)) + (((int)blockIdx.y) * 512)) + (hh_inner_outer * 256)) + (((int)threadIdx.y) * 64)) + (ww_inner_outer * 16)) + ((int)threadIdx.x)) + 32)] = Output_local[1];
        Output[(((((((((nn_inner_outer * 32768) + (((int)threadIdx.z) * 16384)) + (((int)blockIdx.z) * 4096)) + (((int)blockIdx.y) * 512)) + (hh_inner_outer * 256)) + (((int)threadIdx.y) * 64)) + (ww_inner_outer * 16)) + ((int)threadIdx.x)) + 1024)] = Output_local[2];
        Output[(((((((((nn_inner_outer * 32768) + (((int)threadIdx.z) * 16384)) + (((int)blockIdx.z) * 4096)) + (((int)blockIdx.y) * 512)) + (hh_inner_outer * 256)) + (((int)threadIdx.y) * 64)) + (ww_inner_outer * 16)) + ((int)threadIdx.x)) + 1056)] = Output_local[3];
        Output[(((((((((nn_inner_outer * 32768) + (((int)threadIdx.z) * 16384)) + (((int)blockIdx.z) * 4096)) + (((int)blockIdx.y) * 512)) + (hh_inner_outer * 256)) + (((int)threadIdx.y) * 64)) + (ww_inner_outer * 16)) + ((int)threadIdx.x)) + 2048)] = Output_local[4];
        Output[(((((((((nn_inner_outer * 32768) + (((int)threadIdx.z) * 16384)) + (((int)blockIdx.z) * 4096)) + (((int)blockIdx.y) * 512)) + (hh_inner_outer * 256)) + (((int)threadIdx.y) * 64)) + (ww_inner_outer * 16)) + ((int)threadIdx.x)) + 2080)] = Output_local[5];
        Output[(((((((((nn_inner_outer * 32768) + (((int)threadIdx.z) * 16384)) + (((int)blockIdx.z) * 4096)) + (((int)blockIdx.y) * 512)) + (hh_inner_outer * 256)) + (((int)threadIdx.y) * 64)) + (ww_inner_outer * 16)) + ((int)threadIdx.x)) + 3072)] = Output_local[6];
        Output[(((((((((nn_inner_outer * 32768) + (((int)threadIdx.z) * 16384)) + (((int)blockIdx.z) * 4096)) + (((int)blockIdx.y) * 512)) + (hh_inner_outer * 256)) + (((int)threadIdx.y) * 64)) + (ww_inner_outer * 16)) + ((int)threadIdx.x)) + 3104)] = Output_local[7];
      }
    }
  }
}


void Conv2dCpFusedNchwKernelLauncher(const float* U, const float* K0,
    const float* K1, const float* K2, float* V){

  dim3 gridDim0(1, 2, 4);
  dim3 blockDim0(16, 4, 2);

  default_function_kernel0<<<gridDim0, blockDim0>>>(U, K0, K1, K2, V);
  cudaDeviceSynchronize();

}

#endif
