#include "../Tensor.cuh"
#include "../conv.cuh"
#include "../Utils.cuh"

int main() {

  size_t PROFCOUNT = 10000;
  int  x = 32;
  auto U = random_fill({ 1, 1, x, x }, 0, 1);
  auto K = random_fill({ 1, 1, 3, 3 }, 0, 1);
  auto padU = padNCHW(U, 1);

  for (size_t i = 0; i < PROFCOUNT; ++i) {
    auto Full_gpu = conv2d_full_gpu(padU, K);
  }
}
