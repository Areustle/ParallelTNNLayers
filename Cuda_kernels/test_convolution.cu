#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "cudnn_full_conv2d.h"
#include "doctest/doctest.h"
#include <cstdlib>
#include <cudnn.h>
#include <random>

TEST_CASE("Check cudnn full convolution") {

  const size_t dN = 1, dC = 16, dH = 32, dW = 32, dF = 16, dKH = 3, dKW = 3;
  std::random_device               rd;
  std::mt19937                     gen(rd());
  std::uniform_real_distribution<> dis(-1.0, 1.0);

  auto random_fill = [&dis, &gen](size_t len, float* A) {
    for (size_t i = 0; i < len; ++i)
      A[i] = dis(gen);
  };

  size_t len_input  = (dN * dC * dH * dW);
  size_t len_output = (dN * dC * dH * dW);
  size_t len_kernel = (dC * dF * dKH * dKW);

  float* cpu_input  = (float*)std::malloc(len_input * sizeof(float));
  float* cpu_output = (float*)std::malloc(len_output * sizeof(float));
  float* cpu_kernel = (float*)std::malloc(len_kernel * sizeof(float));
  float* gpu_input;
  float* gpu_output;
  float* gpu_kernel;

  random_fill(len_input, cpu_input);
  random_fill(len_output, cpu_output);
  random_fill(len_kernel, cpu_kernel);

  cudaMalloc(&gpu_input, len_input * sizeof(float));
  cudaMalloc(&gpu_output, len_output * sizeof(float));
  cudaMalloc(&gpu_kernel, len_kernel * sizeof(float));
  cudaMemcpy(
      gpu_input, cpu_input, len_input * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_output,
             cpu_output,
             len_input * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_kernel,
             cpu_kernel,
             len_input * sizeof(float),
             cudaMemcpyHostToDevice);

  CudnnConv2d cudcon = CudnnConv2d(dN, dC, dH, dW, dF, dKH, dKW);

  CHECK(cudcon.conv2d(gpu_input, gpu_kernel, gpu_output) ==)

  std::free(cpu_input);
  std::free(cpu_output);
  std::free(cpu_kernel);
  cudaFree(gpu_input);
  cudaFree(gpu_output);
  cudaFree(gpu_kernel);
}
