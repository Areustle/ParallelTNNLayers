#include "../conv.cuh"

int main() {

  float* U;
  float* V;
  float* K;

  unsigned N   = 1;
  unsigned C   = 16;
  unsigned H   = 32;
  unsigned W   = 32;
  unsigned fK  = 16;
  unsigned fH  = 3;
  unsigned fW  = 3;
  unsigned pad = 1;

  cudaMalloc(&U, (N * C * (H + 2 * pad) * (W + 2 * pad)) * sizeof(float));
  cudaMalloc(&K, (fK * C * fH * fW) * sizeof(float));
  cudaMalloc(&V, (N * fK * H * W) * sizeof(float));

  cuda_conv2d_full_gpu(U, N, C, H, W, pad, K, fK, C, fH, fW, V);

  cudaFree(U);
  cudaFree(V);
  cudaFree(K);
}
