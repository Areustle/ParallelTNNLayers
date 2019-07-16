
#include "../Tensor.h"
#include "../cp4Conv2d.h"

int main() {
  const Tensor Input = {1, 16, 32, 32};
  const Tensor Filter = {16, 16, 3, 3};
  auto out = conv2d_full_gpu(Input, Filter);
}
