#include "Tensor.h"
#include "manual.h"
#include <iostream>

__global__ void default_function_kernel0(const float *__restrict__ Data,
                                         const float *__restrict__ K0,
                                         const float *__restrict__ K1,
                                         const float *__restrict__ K2,
                                         const float *__restrict__ K3,
                                         float *__restrict__ Output) {

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
            pad_temp_shared[((((int)threadIdx.z) * 17) + ((int)threadIdx.x))] =
                (((((((1 - ((((((int)threadIdx.z) * 17) + ((int)threadIdx.x)) %
                             68) /
                            34)) -
                      rh_outer) <= (((int)blockIdx.y) * 2)) &&
                    ((((int)blockIdx.y) * 2) <
                     ((33 - ((((((int)threadIdx.z) * 17) + ((int)threadIdx.x)) %
                              68) /
                             34)) -
                      rh_outer))) &&
                   (1 <=
                    (((((int)threadIdx.z) * 17) + ((int)threadIdx.x)) % 34))) &&
                  ((((((int)threadIdx.z) * 17) + ((int)threadIdx.x)) % 34) <
                   33))
                     ? Data[(
                           ((((((((((((int)threadIdx.z) * 17) +
                                    ((int)threadIdx.x)) /
                                   272) *
                                  16384) +
                                 (rc_outer * 4096)) +
                                (((((((int)threadIdx.z) * 17) +
                                    ((int)threadIdx.x)) %
                                   272) /
                                  68) *
                                 1024)) +
                               (((int)blockIdx.y) * 64)) +
                              (((((((int)threadIdx.z) * 17) +
                                  ((int)threadIdx.x)) %
                                 68) /
                                34) *
                               32)) +
                             (rh_outer * 32)) +
                            (((((int)threadIdx.z) * 17) + ((int)threadIdx.x)) %
                             34)) -
                           33)]
                     : 0.000000e+00f);
          }
        }
        if (((int)threadIdx.x) < (4 - ((int)threadIdx.z))) {
          if (((int)threadIdx.x) < 1) {
            if ((rc_outer * 4) <
                ((16 - ((int)threadIdx.z)) - ((int)threadIdx.x))) {
              K0_shared[(((int)threadIdx.x) + ((int)threadIdx.z))] =
                  K0[((((rc_outer * 24) + (((int)threadIdx.x) * 6)) +
                       (((int)threadIdx.z) * 6)) +
                      rr_outer)];
            }
          }
        }
        if (((int)threadIdx.x) < (1 - ((int)threadIdx.z))) {
          if (((int)threadIdx.x) < 1) {
            if (((int)threadIdx.x) < ((3 - rh_outer) - ((int)threadIdx.z))) {
              K1_shared[(((int)threadIdx.x) + ((int)threadIdx.z))] =
                  K1[((((((int)threadIdx.x) * 6) + (((int)threadIdx.z) * 6)) +
                       (rh_outer * 6)) +
                      rr_outer)];
            }
          }
        }
        if (((int)threadIdx.x) < (3 - ((int)threadIdx.z))) {
          if (((int)threadIdx.x) < 1) {
            K2_shared[(((int)threadIdx.x) + ((int)threadIdx.z))] =
                K2[(((((int)threadIdx.x) * 6) + (((int)threadIdx.z) * 6)) +
                    rr_outer)];
          }
        }
        if (((int)threadIdx.x) < (16 - ((int)threadIdx.z))) {
          if (((int)threadIdx.x) < 1) {
            K3_shared[(((int)threadIdx.x) + ((int)threadIdx.z))] =
                K3[(((((int)threadIdx.x) * 6) + (((int)threadIdx.z) * 6)) +
                    rr_outer)];
          }
        }
        __syncthreads();
        for (int rc_inner_outer = 0; rc_inner_outer < 4; ++rc_inner_outer) {
          for (int ax2 = 0; ax2 < 2; ++ax2) {
            for (int ax3 = 0; ax3 < 3; ++ax3) {
              pad_temp_shared_local[((ax2 * 3) + ax3)] = pad_temp_shared[(
                  (((rc_inner_outer * 68) + (ax2 * 34)) + ax3) +
                  ((int)threadIdx.x))];
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
              Output_local[hh_c] =
                  (Output_local[hh_c] +
                   ((((pad_temp_shared_local[((hh_c * 3) + rw_inner_inner)] *
                       K0_shared_local[0]) *
                      K1_shared_local[0]) *
                     K2_shared_local[rw_inner_inner]) *
                    K3_shared_local[0]));
            }
          }
        }
      }
    }
  }
  for (int hh_inner_inner_inner = 0; hh_inner_inner_inner < 2;
       ++hh_inner_inner_inner) {
    Output[((((((int)threadIdx.z) * 1024) + (((int)blockIdx.y) * 64)) +
             (hh_inner_inner_inner * 32)) +
            ((int)threadIdx.x))] = Output_local[hh_inner_inner_inner];
  }
}

Tensor static_cp4_conv2d(Tensor const U, Tensor const K0, Tensor const K1,
                         Tensor const K2, Tensor const K3) {

  Tensor V{1, 16, 32, 32};
  dim3 gridDim0(1, 16, 1);
  dim3 blockDim0(32, 1, 16);

  cudaDeviceSynchronize();
  default_function_kernel0<<<gridDim0, blockDim0>>>(
      U.m_data, K0.m_data, K1.m_data, K2.m_data, K3.m_data, V.m_data);
  cudaDeviceSynchronize();

  return V;
}

/* Tensor naive_conv2d(Tensor U, Tensor Kernel0, Tensor Kernel1, */
/* Tensor Kernel2,                     Tensor Kernel3) { */
/*   const size_t N = U.shape[0]; */
/*   const size_t C = U.shape[1]; */
/*   const size_t H = U.shape[2]; */
/*   const size_t W = U.shape[3]; */

/*   const size_t rank = Kernel0.shape[1]; */
/*   const int K0 = Kernel0.shape[0]; */
/*   const int K1 = Kernel1.shape[0]; */
/*   const int K2 = Kernel2.shape[0]; */
/*   const int K3 = Kernel3.shape[0]; */

/*   Tensor Out({K0, K1, K2, K3}); */

/*   // clang-format off */
/*   for (int r = 0; r < rank; ++r) { */
/*     for (int n = 0; n < N; ++n) */
/*     for (int c = 0; c < C; ++c) */
/*     for (int h = 0; h < H; ++h) */
/*     for (int w = 0; w < W; ++w) */
/*     for (int a = 0; a < K0; ++a) */
/*     for (int b = 0; b < K1; ++b) */
/*     for (int c = 0; c < K2; ++c) */
/*     for (int d = 0; d < K3; ++d) */
/*       r; */
/*   } */
/*   // clang-format on */

/*   return Out; */
/* } */

#define DOCTEST_CONFIG_IMPLEMENTATION_IN_DLL
#include "../external/doctest/doctest.h"
#include "Utils.h"
#include "cudnnConv2d.h"

TEST_CASE("Utils test") {

  auto U = random_fill({1, 16, 32, 32}, 0, 1);
  auto K0 = random_fill({16, 6}, 0, 1);
  auto K1 = random_fill({16, 6}, 0, 1);
  auto K2 = random_fill({3, 6}, 0, 1);
  auto K3 = random_fill({3, 6}, 0, 1);
  auto K = cp4recom(K0, K1, K2, K3);
  auto V = nn_conv2d(U, K);
  auto W = static_cp4_conv2d(U, K0, K2, K3, K1);

  REQUIRE(V.size() == W.size());

  for (int i = 0; i < V.size(); ++i)
    REQUIRE(V[i] == W[i]);
}
