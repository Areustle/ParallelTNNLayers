
__constant__ float carray[1 << 13];

__global__ void conv2d_full_kernel(const float *__restrict__ Input,
                                   const int C,
                                   const int K,
                                   const int R,
                                   const int FRCenter,
                                   const int S,
                                   const int FSCenter,
                                   float *__restrict__ Out) {

  const int n = blockIdx.x;

  const int H = gridDim.y;
  const int h = blockIdx.y;

  const int W = gridDim.z;
  const int w = blockIdx.z;

  // clang-format off
  for (int k = 0; k < K; ++k){
    float sum = 0.0f;
    for (int c = 0; c < C; ++c)
    for (int r = 0; r < R; ++r)
    for (int s = 0; s < S; ++s){

      const int hIdx = h + (r - FRCenter);
      const int wIdx = w + (s - FSCenter);
      const int fidx = k*C*R*S + c*R*S + r*S + s;
      const float fil = carray[fidx];

      if(hIdx >= 0 && hIdx < H && wIdx >= 0 && wIdx < W){
            sum += Input[n*C*H*W + c*H*W + hIdx*W + wIdx]
            * fil;
      }

    }
  Out[n*C*H*W + k*H*W + h*W + w] = sum;
  }
  // clang-format on
}

int main() {

  size_t PROFCOUNT = 1000;


  const int N = 1, C = 16, H = 32, W = 32;
  const int K = 16, R = 3, S = 3;

  const int FRCenter = R / 2;
  const int FSCenter = S / 2;

  float *In;
  float *Out;
  float  Fil[K * C * R * S];

  cudaMalloc(&In, (N * C * H * W) * sizeof(float));
  cudaMemcpyToSymbol(carray, Fil, (K * C * R * S) * sizeof(float));
  cudaMalloc(&Out, (N * C * H * W) * sizeof(float));

  dim3 gridDim0(N, H, W);
  dim3 blockDim0(1, 1, 1);

  for (size_t i = 0; i < PROFCOUNT; ++i) {
    conv2d_full_kernel<<<gridDim0, blockDim0>>>(
        In, C, K, R, FRCenter, S, FSCenter, Out);
    cudaDeviceSynchronize();
  }

  cudaFree(In);
  cudaFree(Fil);
  cudaFree(Out);
}
