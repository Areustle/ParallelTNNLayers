#include "../cp4Conv2d.cuh"

int main() {

  float* In;
  float* Out;
  float*  FilterK;
  float*  FilterC;
  float*  FilterW;
  float*  FilterH;

  unsigned N     = 1;
  unsigned C     = 16;
  unsigned H     = 32;
  unsigned W     = 32;
  unsigned fK    = 16;
  unsigned fH    = 3;
  unsigned fW    = 3;
  unsigned fRank = 1;
  unsigned pad   = 1;


  cudaMalloc(&In, (N * C * (H + 2) * (W + 2)) * sizeof(float));
  cudaMalloc(&FilterK, (fK * fRank) * sizeof(float));
  cudaMalloc(&FilterC, (C * fRank) * sizeof(float));
  cudaMalloc(&FilterH, (fH * fRank) * sizeof(float));
  cudaMalloc(&FilterW, (fW * fRank) * sizeof(float));
  cudaMalloc(&Out, (N * fK * H * W) * sizeof(float));

  cuda_conv2d_cp4_gpu(In,
                      N,
                      C,
                      H,
                      W,
                      pad,
                      FilterK,
                      FilterC,
                      FilterH,
                      FilterW,
                      fRank,
                      fK,
                      C,
                      fH,
                      fW,
                      Out);

  cudaFree(In);
  cudaFree(FilterK);
  cudaFree(FilterC);
  cudaFree(FilterH);
  cudaFree(FilterW);
  cudaFree(Out);
}
