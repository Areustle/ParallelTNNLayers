#define DOCTEST_CONFIG_IMPLEMENTATION_IN_DLL
#include "../../external/doctest/doctest.h"
#include "../Tensor.cuh"
#include <random>


using namespace std;

TEST_CASE("Testing the Tensor Class") {

  random_device               rd;
  mt19937                     gen(rd());
  uniform_real_distribution<> dis(-1.0, 1.0);


  Tensor ten = { 40, 1, 1 };
  for (size_t i = 0; i < ten.size(); ++i) ten.m_data[i] = dis(gen);
  Tensor input(ten);

  CHECK(ten.size() == 40);
  CHECK(ten.order() == 3);
  CHECK(input.m_data[0] == doctest::Approx(ten.m_data[0]).epsilon(1e-3));

  for (int i = 0; i < ten.size(); ++i) {
    REQUIRE(input.m_data[i] == doctest::Approx(ten.m_data[i]).epsilon(1e-3));
  }
}
