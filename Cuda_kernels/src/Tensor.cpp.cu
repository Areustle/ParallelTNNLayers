#include "Tensor.h"

#include <numeric>

using namespace std;

Tensor::Tensor(std::initializer_list<int> l)
    : shape(l) {
  const int len =
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
  cudaMallocManaged(&m_data, len * sizeof(float));
  cudaMemset(&m_data, 0, len);
  cudaDeviceSynchronize();
}


Tensor::Tensor(Tensor const& other)
    : shape(other.shape) {
  const int len =
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
  cudaMallocManaged(&m_data, len * sizeof(float));
  cudaMemcpy(
      m_data, other.m_data, len * sizeof(float), cudaMemcpyDeviceToDevice);
  cudaDeviceSynchronize();
}


Tensor& Tensor::operator=(const Tensor& other) {
  if (this == &other)
    return *this;
  if (this->size() != other.size()) {
    delete[] m_data;
    cudaMallocManaged(&m_data, other.size() * sizeof(float));
  }
  cudaMemcpy(m_data,
             other.m_data,
             other.size() * sizeof(float),
             cudaMemcpyDeviceToDevice);
  cudaDeviceSynchronize();
  return *this;
}


Tensor::~Tensor() {
  cudaDeviceSynchronize();
  cudaFree(m_data);
}

size_t Tensor::size() const {
  return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
}


#define DOCTEST_CONFIG_IMPLEMENTATION_IN_DLL
#include "doctest.h"
#include <random>

using namespace std;

TEST_CASE("Testing the Tensor Class") {

  random_device               rd;
  mt19937                     gen(rd());
  uniform_real_distribution<> dis(-1.0, 1.0);

  auto random_fill = [&dis, &gen](Tensor A) {
    for (size_t i = 0; i < A.size(); ++i)
      A[i] = dis(gen);
  };

  Tensor ten = { 40, 1, 1 };
  random_fill(ten);
  Tensor input(ten);

  CHECK(input[0] == doctest::Approx(ten[0]).epsilon(1e-3));

  for (int i = 0; i < ten.size(); ++i) {
    REQUIRE(input[i] == doctest::Approx(ten[i]).epsilon(1e-3));
  }
}
