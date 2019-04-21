#if GOOGLE_CUDA
#define EIGEN_USE_GPU

__global__
void NMode32Kernel(const float* A, const int I, const int J, const int S,
                   const float* B, const int R,
                   float* C){
  int32_t iA = blockIdx.x * blockDim.x + threadIdx.x;
  if (iA >= I) {
    return;
  }

  for (int32_t jA = 0; jA < J; jA++) {
    int32_t p2 = iA * J + jA;
    for (int32_t sA = 0; sA < S; sA++) {
      float ts = A[p2 * S + sA];
      for (int32_t rB = 0; rB < R; rB++) {
        int32_t pB2 = sA * R + rB;
        int32_t pC3 = p2 * R + rB;
        C[pC3] += ts * B[pB2];
      }
    }
  }
}

void NMode32KernelLauncher(const float* A, const int I, const int J, const int S,
                   const float* B, const int R,
                   float* C) {
  NMode32Kernel<<<(I + 255) / 256, 256>>>(A, I, J, S, B, R, C);
  cudaDeviceSynchronize();
}

#endif
