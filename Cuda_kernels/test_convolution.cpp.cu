#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

/* #include "cudnn_full_conv2d.h" */
#include "CudaAllocator.h"
#include "Tensor.h"
#include "doctest/doctest.h"
#include <cstdlib>
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

  Tensor<> cpu_input(dN, dC, dH, dW);
  Tensor<> cpu_output(cpu_input);
  Tensor<> cpu_kernel = Tensor<>(dC, dF, dKH, dKW);

  random_fill(len_input, cpu_input.data);
  random_fill(len_output, cpu_output.data);
  random_fill(len_kernel, cpu_kernel.data);

  Tensor<> gpu_input(cpu_input);
  Tensor<> gpu_output(cpu_output);
  Tensor<> gpu_kernel = cpu_kernel;

  CHECK(cpu_input.data[0] == gpu_input.data[0]);
  CHECK(cpu_input.data[0] == doctest::Approx(gpu_input.data[0]).epsilon(1e-3));

  for (int i=0; i<cpu_input.len; ++i){
    CHECK(cpu_input.data[i] == doctest::Approx(gpu_input.data[i]).epsilon(1e-3));
  }
}
