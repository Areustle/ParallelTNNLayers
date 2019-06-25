#include "cpu_conv2d.h"

void cpu_imp::conv2d(float*       U, //[N,C,H,W]
            float*       K, //[C,F,KH,KW]
            float*       V, //[N,C,H,W]
            const size_t dN  ,
            const size_t dC  ,
            const size_t dH  ,
            const size_t dW  ,
            const size_t dF  ,
            const size_t dKH ,
            const size_t dKW ){
  for (int n=0; n<dN; ++n)
  for (int c=0; c<dC; ++c)
  for (int h=0; h<dH; ++h)
  for (int w=0; w<dW; ++w)
    V[n + c*dN + h*dN*dC + w*dN*dC*dH] = 0 ;

  for (int n=0; n<dN; ++n)
  for (int c=0; c<dC; ++c)
  for (int h=0; h<dH; ++h)
  for (int w=0; w<dW; ++w)
  for (int f=0; f<dF; ++f)
  for (int kh=0; kh<dKH; ++kh)
  for (int kw=0; kw<dKW; ++kw)
    V[n*dC*dW*dH + c*dW*dH + h*dW + w] +=
      K[n*dF*dKW*dKH + f*dKW*dKW + kh*dKW + kw]
      * U[n*dC*dW*dH + c*dW*dH + (h+kh)*dW + (w+kw)];

}
