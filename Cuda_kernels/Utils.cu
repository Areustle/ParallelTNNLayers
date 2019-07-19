#include "Utils.cuh"

#include <random>

using namespace std;

Tensor random_fill(std::initializer_list<int> lst, float lo, float hi) {

  random_device               rd;
  mt19937                     gen(rd());
  uniform_real_distribution<> dis(lo, hi);

  Tensor A(lst);

  for (size_t i = 0; i < A.size(); ++i) A.m_data[i] = dis(gen);

  return A;
};

Tensor
cp4recom(Tensor FilterK, Tensor FilterC, Tensor FilterR, Tensor FilterS) {
  const size_t rank = FilterK.shape[1];
  const int    FK   = FilterK.shape[0];
  const int    FC   = FilterC.shape[0];
  const int    FR   = FilterR.shape[0];
  const int    FS   = FilterS.shape[0];
  Tensor       Out  = { FK, FC, FR, FS };

  // clang-format off
  for (int a = 0; a < FK; ++a)
  for (int b = 0; b < FC; ++b)
  for (int c = 0; c < FR; ++c)
  for (int d = 0; d < FS; ++d)
  for (int r = 0; r < rank; ++r)
    Out.m_data[a*FC*FR*FS + b*FR*FS + c*FS + d]
      += FilterK.m_data[a*rank + r]
       * FilterC.m_data[b*rank + r]
       * FilterR.m_data[c*rank + r]
       * FilterS.m_data[d*rank + r];
  // clang-format on

  return Out;
}


Tensor padNCHW(Tensor In, int pad = 1) {
  int N  = In.shape[0];
  int C  = In.shape[1];
  int iH = In.shape[2];
  int iW = In.shape[3];

  int oH = iH + (2 * pad);
  int oW = iW + (2 * pad);

  Tensor Out = { N, C, oH, oW };

  // clang-format off
  for (int n=0; n<N; ++n)
  for (int c=0; c<C; ++c)
  for (int h=0; h<oH; ++h)
  for (int w=0; w<oW; ++w)
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
