#include "Tensor.h"

using namespace std;

Tensor::Tensor(size_t N, size_t C, size_t H, size_t W)
    : N(N)
    , C(C)
    , H(H)
    , W(W)
    , len(N * C * H * W) {
  cudaMallocManaged(&m_data, len * sizeof(float));
  cudaMemset(&m_data, 0, len);
  cudaDeviceSynchronize();
}

Tensor::Tensor(Tensor const& other)
    : N(other.N)
    , C(other.C)
    , H(other.H)
    , W(other.W)
    , len(other.len) {
  cudaMallocManaged(&m_data, len * sizeof(float));
  cudaMemcpy(
      m_data, other.m_data, len * sizeof(float), cudaMemcpyDeviceToDevice);
  cudaDeviceSynchronize();
}

Tensor& Tensor::operator=(Tensor const& other) {
  if (this == &other)
    return *this;
  if (len != other.len) {
    delete[] m_data;
    len = other.len;
    cudaMallocManaged(&m_data, len * sizeof(float));
  }
  cudaMemcpy(
      m_data, other.m_data, len * sizeof(float), cudaMemcpyDeviceToDevice);
  cudaDeviceSynchronize();
  return *this;
}

Tensor::~Tensor() {
  cudaDeviceSynchronize();
  cudaFree(m_data);
}


#define DOCTEST_CONFIG_IMPLEMENTATION_IN_DLL
#include "doctest.h"
#include <random>

using namespace std;

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

  random_fill(ten.size(), ten);

  Tensor input(ten);

  CHECK(input[0] == doctest::Approx(ten[0]).epsilon(1e-3));

  for (int i = 0; i < ten.size(); ++i) {
    REQUIRE(input[i] == doctest::Approx(ten[i]).epsilon(1e-3));
  }
}
