#include "Cuda_kernels/Tensor.h"

using namespace std;

Tensor::Tensor(size_t N, size_t C, size_t H, size_t W)
    : N(N)
    , C(C)
    , H(H)
    , W(W)
    , size(N * C * H * W) {
  cudaMallocManaged(&m_data, size * sizeof(float));
  cudaMemset(&m_data, 0, size);
  cudaDeviceSynchronize();
}

Tensor::Tensor(Tensor const& other)
    : N(other.N)
    , C(other.C)
    , H(other.H)
    , W(other.W)
    , size(other.size) {
  cudaMallocManaged(&m_data, size * sizeof(float));
  cudaMemcpy(m_data, other.m_data, size * sizeof(float), cudaMemcpyDeviceToDevice);
  cudaDeviceSynchronize();
}

Tensor& Tensor::operator=(Tensor const& other) {
  if (this == &other)
    return *this;
  if (size != other.size) {
    delete[] m_data;
    size = other.size;
    cudaMallocManaged(&m_data, size * sizeof(float));
  }
  cudaMemcpy(m_data, other.m_data, size * sizeof(float), cudaMemcpyDeviceToDevice);
  cudaDeviceSynchronize();
  return *this;
}

Tensor::~Tensor() {
  cudaDeviceSynchronize();
  cudaFree(m_data);
}


/******************************************************************************
* Test Code Below.
******************************************************************************/
#define DOCTEST_CONFIG_IMPLEMENTATION_IN_DLL
#include "doctest/doctest.h"
#include <random>


TEST_CASE("Testing the Tensor Class") {

  const size_t  dN = 1, dC = 1, dH = 1, dW = 4000; //, dF = 16, dKH = 3, dKW = 3;
  random_device rd;
  mt19937       gen(rd());
  uniform_real_distribution<> dis(-1.0, 1.0);

  auto random_fill = [&dis, &gen](size_t len, Tensor A) {
    for (size_t i = 0; i < len; ++i)
      A[i] = dis(gen);
  };

  Tensor ten(dN, dC, dH, dW);
  random_fill(ten.size, ten);

  Tensor input(ten);

  CHECK(input[0] == doctest::Approx(ten[0]).epsilon(1e-3));

  for (int i = 0; i < ten.size; ++i) {
    REQUIRE(input[i] == doctest::Approx(ten[i]).epsilon(1e-3));
  }
}
