#include <cstddef>

void cudnn_imp::conv2d( float*       U,
                        void*        K,
                        float*       V,
                        const size_t dN  = 1,
                        const size_t dC  = 16,
                        const size_t dH  = 32,
                        const size_t dW  = 32,
                        const size_t dF  = 16,
                        const size_t dKH = 3,
                        const size_t dKW = 3 );
