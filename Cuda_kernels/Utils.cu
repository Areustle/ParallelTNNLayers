#include "Utils.cuh"

#include <random>

using namespace std;

Tensor random_fill(std::initializer_list<unsigned> lst, float lo, float hi) {

  random_device               rd;
  mt19937                     gen(rd());
  uniform_real_distribution<> dis(lo, hi);

  Tensor A(lst);

  for (size_t i = 0; i < A.size(); ++i) A.m_data[i] = dis(gen);

  return A;
};

Tensor
cp4recom(Tensor FilterK, Tensor FilterC, Tensor FilterR, Tensor FilterS) {
  const unsigned rank = FilterK.shape[1];
  const unsigned FK   = FilterK.shape[0];
  const unsigned FC   = FilterC.shape[0];
  const unsigned FR   = FilterR.shape[0];
  const unsigned FS   = FilterS.shape[0];
  Tensor         Out  = { FK, FC, FR, FS };

  // clang-format off
  for (unsigned a = 0; a < FK; ++a)
  for (unsigned b = 0; b < FC; ++b)
  for (unsigned c = 0; c < FR; ++c)
  for (unsigned d = 0; d < FS; ++d)
  for (unsigned r = 0; r < rank; ++r)
    Out.m_data[a*FC*FR*FS + b*FR*FS + c*FS + d]
      += FilterK.m_data[a*rank + r]
       * FilterC.m_data[b*rank + r]
       * FilterR.m_data[c*rank + r]
       * FilterS.m_data[d*rank + r];
  // clang-format on

  return Out;
}


Tensor padNCHW(Tensor In, unsigned pad = 1) {
  unsigned N  = In.shape[0];
  unsigned C  = In.shape[1];
  unsigned iH = In.shape[2];
  unsigned iW = In.shape[3];

  unsigned oH = iH + (2 * pad);
  unsigned oW = iW + (2 * pad);

  Tensor Out = { N, C, oH, oW };

  // clang-format off
  for (unsigned n=0; n<N; ++n)
  for (unsigned c=0; c<C; ++c)
  for (unsigned h=0; h<oH; ++h)
  for (unsigned w=0; w<oW; ++w)
    if(h>=pad && h<=iH && w>=pad && w<=iW){
      Out.m_data[n*C*oH*oW + c*oH*oW + h*oW + w]
        = In.m_data[n*C*iH*iW + c*iH*iW + (h-pad)*iW + (w-pad)];
    }
    else{
      Out.m_data[n*C*oH*oW + c*oH*oW + h*oW + w] = 0;
    }
  // clang-format on

  return Out;
}
