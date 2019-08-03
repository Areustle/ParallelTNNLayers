#include "../NVConv2d.cuh"

int main() {

  float* In;
  float* Out;
  float* Filter;

  unsigned N     = 1;
  unsigned C     = 16;
  unsigned H     = 32;
  unsigned W     = 32;
  unsigned fK    = 16;
  unsigned fH    = 3;
  unsigned fW    = 3;

  cudaMalloc(&In, (N * C * H * W));
  cudaMalloc(&Filter, fK * C * fH * fW);
  cudaMalloc(&Out, (N * fK * H * W));

  float milliseconds = NV::conv2d_forward_gpu(In,
                                              N,
                                              C,
                                              H,
                                              W,
                                              Filter,
                                              fK,
                                              fH,
                                              fW,
                                              Out);

  printf("Elapsed Time: %f us \n", milliseconds * 1e3);

  cudaFree(In);
  cudaFree(Filter);
  cudaFree(Out);
}
