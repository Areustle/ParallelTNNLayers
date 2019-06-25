#include <cstddef>

namespace cpu_imp {
  void conv2d( float*       U,
               float*       K,
               float*       V,
               const size_t dN  = 1,
               const size_t dC  = 16,
               const size_t dH  = 32,
               const size_t dW  = 32,
               const size_t dF  = 16,
               const size_t dKH = 3,
               const size_t dKW = 3 );
}
