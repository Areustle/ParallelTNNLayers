#include "../cp4Conv2d.cuh"
#include <iostream>

int main() {

  float* In;
  float* Out;
  float* FilterK;
  float* FilterC;
  float* FilterW;
  float* FilterH;

  unsigned N     = 1;
  unsigned C     = 16;
  unsigned H     = 32;
  unsigned W     = 32;
  unsigned fK    = 16;
  unsigned fH    = 3;
  unsigned fW    = 3;
  unsigned fRank = 1;
  unsigned pad   = 1;


  cudaMalloc(&In, (N * C * (H + 2) * (W + 2)));
  cudaMalloc(&FilterK, (fK * fRank));
  cudaMalloc(&FilterC, (C * fRank));
  cudaMalloc(&FilterH, (fH * fRank));
  cudaMalloc(&FilterW, (fW * fRank));
  cudaMalloc(&Out, (N * fK * H * W));

  float milliseconds = cuda_conv2d_cp4_gpu(In,
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

  std::cout << "Elapsed Time: " << milliseconds * 1e3 << " \u03BCs "
            << std::endl;
  /* printf("Effective Bandwidth (GB/s): %fn", N*4*3/milliseconds/1e6); */


  cudaFree(In);
  cudaFree(FilterK);
  cudaFree(FilterC);
  cudaFree(FilterH);
  cudaFree(FilterW);
  cudaFree(Out);
}
