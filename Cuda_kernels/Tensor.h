#include <cstddef>

class Tensor
{
private:
  size_t N;
  size_t C;
  size_t H;
  size_t W;

  float* data;

public:
  Tensor(size_t N, size_t C, size_t H, size_t W);
  ~Tensor();
  Tensor(const Tensor& other) = delete;
  Tensor& operator=(const Tensor& other) = delete;
};
