#include "../Tensor.cuh"


__global__ void kernel(const float* __restrict__ Input){

  extern __shared__ float shared_mem[];

  shared_mem[threadIdx.x] = Input[threadIdx.x];

}


int main() {

  float* In;
  cudaMalloc(&In, (4096*sizeof(float)));
  kernel<<<256,4096/256, 256>>>(In);
  cudaFree(In);

}
