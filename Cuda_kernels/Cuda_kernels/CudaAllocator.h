#ifndef CUDAALLOCATOR_H
#define CUDAALLOCATOR_H

#include <cstddef>
#include <memory>

class CudaAllocator
{
private:
public:
  typedef float value_type;

  CudaAllocator() noexcept {}

  CudaAllocator(const CudaAllocator&) noexcept {}

  static float* allocate(std::size_t n); 

  static void deallocate(float* p, std::size_t n); // { cudaFree(p); }
};

#endif /* CUDAALLOCATOR_H */
