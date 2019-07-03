#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "Tensor.h"
#include "doctest/doctest.h"
#include <iostream>
#include <random>

using namespace std;

/* int main() { */
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
