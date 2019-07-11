#define DOCTEST_CONFIG_IMPLEMENTATION_IN_DLL
#include "../../external/doctest/doctest.h"
#include "../Tensor.h"
#include <random>


using namespace std;

TEST_CASE("Testing the Tensor Class") {

  random_device               rd;
  mt19937                     gen(rd());
  uniform_real_distribution<> dis(-1.0, 1.0);

  auto r_fill = [&dis, &gen](Tensor A) {
    for (size_t i = 0; i < A.size(); ++i) A[i] = dis(gen);
  };

  Tensor ten = { 40, 1, 1 };
  r_fill(ten);
  /* Tensor input(ten); */

  /* CHECK(ten.size() == 40); */
  /* CHECK(ten.order() == 3); */
  /* CHECK(input[0] == doctest::Approx(ten[0]).epsilon(1e-3)); */

  /* for (int i = 0; i < ten.size(); ++i) { */
  /*   REQUIRE(input[i] == doctest::Approx(ten[i]).epsilon(1e-3)); */
  /* } */
}
