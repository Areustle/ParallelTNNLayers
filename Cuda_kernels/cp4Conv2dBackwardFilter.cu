#include "cp4Conv2dBackwardData.cuh"
#include <iostream>
#include <stdlib.h>

using namespace std;

// Simple cuda error checking macro
#define ErrChk(ans) \
  { CudaAssert((ans), __FILE__, __LINE__); }
inline void
CudaAssert(cudaError_t code, const char* file, int line, bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(
        stderr, "CudaAssert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
float cp4_conv2d_backward_filter_full_gpu(tensor_shape s,
                                       float* FF,
                                       const float* In,
                                       float* SHIn,
                                       const float* dLdO,
                                       float* SHOut,
                                       unsigned     PROFCOUNT) {
  const unsigned sH = s.H+2*s.pad;
  const unsigned sW = s.W+2*s.pad;

  for (int n = 0; n < s.N; ++n)
  for (int h = 0; h < s.H; ++h)
  for (int w = 0; w < s.W; ++w){

    for (int t = 0; t < s.T; ++t){
      SHOut[n*s.T*sH*sW + t*sH*sW + h*sW + w]
            = (h >= s.pad
                && h < s.H+s.pad
                && w >= s.pad
                && w < s.W+s.pad)
            ? dLdO[n*s.T*s.H*s.W + t*s.H*s.W + (h-s.pad)*s.W + (w-s.pad)]
            : 0.0f;
    }

    for (int c = 0; c < s.C; ++c){
      SHIn[n*s.C*sH*sW + c*sH*sW + h*sW + w]
        = (h >= s.pad
            && h < s.H+s.pad
            && w >= s.pad
            && w < s.W+s.pad)
        ? In[n*s.C*s.H*s.W + c*s.H*s.W + (h-s.pad)*s.W + (w-s.pad)]
        : 0.0f;

    }
  }

  for (int t = 0; t < s.T; ++t)
  for (int c = 0; c < s.C; ++c)
  for (int y = 0; y < s.Y; ++y)
  for (int x = 0; x < s.X; ++x)
  for (int n = 0; n < s.N; ++n)
  for (int h = 0; h < s.H; ++h)
  for (int w = 0; w < s.W; ++w)
    FF[t*s.C*s.Y*s.X + c*s.Y*s.X + y*s.X +x] 
      += dLdO[n*s.T*s.H*s.W + t*s.H*s.W + h*s.W +w]
       * SHIn[n*s.C*sH*sW + c*sH*sW + (h+y)*sW + (w+x)];

  return 0;
}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
float cp4_conv2d_backward_filter_t_gpu(tensor_shape s,
                                       float* FT,
                                       const float* In,
                                       float* SHIn,
                                       const float* dLdO,
                                       float* SHOut,
                                       const float* FC,
                                       const float* FY,
                                       const float* FX,
                                       unsigned     PROFCOUNT) {
  const unsigned sH = s.H+2*s.pad;
  const unsigned sW = s.W+2*s.pad;

  for (int n = 0; n < s.N; ++n)
  for (int h = 0; h < s.H; ++h)
  for (int w = 0; w < s.W; ++w){

    for (int t = 0; t < s.T; ++t){
      SHOut[n*s.T*sH*sW + t*sH*sW + h*sW + w]
            = (h >= s.pad
                && h < s.H+s.pad
                && w >= s.pad
                && w < s.W+s.pad)
            ? dLdO[n*s.T*s.H*s.W + t*s.H*s.W + (h-s.pad)*s.W + (w-s.pad)]
            : 0.0f;
    }

    for (int c = 0; c < s.C; ++c){
      SHIn[n*s.C*sH*sW + c*sH*sW + h*sW + w]
        = (h >= s.pad
            && h < s.H+s.pad
            && w >= s.pad
            && w < s.W+s.pad)
        ? In[n*s.C*s.H*s.W + c*s.H*s.W + (h-s.pad)*s.W + (w-s.pad)]
        : 0.0f;

    }
  }

  for (int t = 0; t < s.T; ++t)
  for (int r = 0; r < s.Rank; ++r)
  for (int n = 0; n < s.N; ++n)
  for (int h = 0; h < s.H; ++h)
  for (int w = 0; w < s.W; ++w)
  for (int c = 0; c < s.C; ++c)
  for (int y = 0; y < s.Y; ++y)
  for (int x = 0; x < s.X; ++x)
    FT[t*s.Rank + r] += SHOut[n*s.T*s.H*s.W + t*s.H*s.W + h*s.W +w]
      *SHIn[n*s.C*s.H*s.W + c*s.H*s.W + (h+y)*s.W + (w+x)]
      *FC[c*s.Rank + r]
      *FY[y*s.Rank + r]
      *FX[x*s.Rank + r];

  return 0;
}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
float cp4_conv2d_backward_filter_c_gpu(tensor_shape s,
                                       float* FC,
                                       const float* In,
                                       float* SHIn,
                                       const float* dLdO,
                                       float* SHOut,
                                       const float* FT,
                                       const float* FY,
                                       const float* FX,
                                       unsigned     PROFCOUNT) {

  const unsigned sH = s.H+2*s.pad;
  const unsigned sW = s.W+2*s.pad;

  for (int n = 0; n < s.N; ++n)
  for (int h = 0; h < s.H; ++h)
  for (int w = 0; w < s.W; ++w){

    for (int t = 0; t < s.T; ++t){
      SHOut[n*s.T*sH*sW + t*sH*sW + h*sW + w]
            = (h >= s.pad
                && h < s.H+s.pad
                && w >= s.pad
                && w < s.W+s.pad)
            ? dLdO[n*s.T*s.H*s.W + t*s.H*s.W + (h-s.pad)*s.W + (w-s.pad)]
            : 0.0f;
    }

    for (int c = 0; c < s.C; ++c){
      SHIn[n*s.C*sH*sW + c*sH*sW + h*sW + w]
        = (h >= s.pad
            && h < s.H+s.pad
            && w >= s.pad
            && w < s.W+s.pad)
        ? In[n*s.C*s.H*s.W + c*s.H*s.W + (h-s.pad)*s.W + (w-s.pad)]
        : 0.0f;

    }
  }

  for (int c = 0; c < s.C; ++c)
  for (int r = 0; r < s.Rank; ++r)
  for (int n = 0; n < s.N; ++n)
  for (int t = 0; t < s.T; ++t)
  for (int h = 0; h < s.H; ++h)
  for (int w = 0; w < s.W; ++w)
  for (int y = 0; y < s.Y; ++y)
  for (int x = 0; x < s.X; ++x)
    FC[c*s.Rank + r] += SHOut[n*s.T*s.H*s.W + t*s.H*s.W + h*s.W +w]
      *SHIn[n*s.C*s.H*s.W + c*s.H*s.W + (h+y)*s.W + (w+x)]
      *FT[t*s.Rank + r]
      *FY[y*s.Rank + r]
      *FX[x*s.Rank + r];

  return 0;
}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
float cp4_conv2d_backward_filter_y_gpu(tensor_shape s,
                                       float* FY,
                                       const float* In,
                                       float* SHIn,
                                       const float* dLdO,
                                       float* SHOut,
                                       const float* FT,
                                       const float* FC,
                                       const float* FX,
                                       unsigned     PROFCOUNT) {

  const unsigned sH = s.H+2*s.pad;
  const unsigned sW = s.W+2*s.pad;

  for (int n = 0; n < s.N; ++n)
  for (int h = 0; h < s.H; ++h)
  for (int w = 0; w < s.W; ++w){

    for (int t = 0; t < s.T; ++t){
      SHOut[n*s.T*sH*sW + t*sH*sW + h*sW + w]
            = (h >= s.pad
                && h < s.H+s.pad
                && w >= s.pad
                && w < s.W+s.pad)
            ? dLdO[n*s.T*s.H*s.W + t*s.H*s.W + (h-s.pad)*s.W + (w-s.pad)]
            : 0.0f;
    }

    for (int c = 0; c < s.C; ++c){
      SHIn[n*s.C*sH*sW + c*sH*sW + h*sW + w]
        = (h >= s.pad
            && h < s.H+s.pad
            && w >= s.pad
            && w < s.W+s.pad)
        ? In[n*s.C*s.H*s.W + c*s.H*s.W + (h-s.pad)*s.W + (w-s.pad)]
        : 0.0f;

    }
  }

  for (int y = 0; y < s.Y; ++y)
  for (int r = 0; r < s.Rank; ++r)
  for (int n = 0; n < s.N; ++n)
  for (int h = 0; h < s.H; ++h)
  for (int w = 0; w < s.W; ++w)
  for (int c = 0; c < s.C; ++c)
  for (int t = 0; t < s.T; ++t)
  for (int x = 0; x < s.X; ++x)
    FY[y*s.Rank + r] += SHOut[n*s.T*s.H*s.W + t*s.H*s.W + h*s.W +w]
      *SHIn[n*s.C*s.H*s.W + c*s.H*s.W + (h+y)*s.W + (w+x)]
      *FT[t*s.Rank + r]
      *FC[c*s.Rank + r]
      *FX[x*s.Rank + r];

  return 0;
}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
float cp4_conv2d_backward_filter_x_gpu(tensor_shape s,
                                       float* FX,
                                       const float* In,
                                       float* SHIn,
                                       const float* dLdO,
                                       float* SHOut,
                                       const float* FT,
                                       const float* FC,
                                       const float* FY,
                                       unsigned     PROFCOUNT) {

  const unsigned sH = s.H+2*s.pad;
  const unsigned sW = s.W+2*s.pad;

  for (int n = 0; n < s.N; ++n)
  for (int h = 0; h < s.H; ++h)
  for (int w = 0; w < s.W; ++w){

    for (int t = 0; t < s.T; ++t){
      SHOut[n*s.T*sH*sW + t*sH*sW + h*sW + w]
            = (h >= s.pad
                && h < s.H+s.pad
                && w >= s.pad
                && w < s.W+s.pad)
            ? dLdO[n*s.T*s.H*s.W + t*s.H*s.W + (h-s.pad)*s.W + (w-s.pad)]
            : 0.0f;
    }

    for (int c = 0; c < s.C; ++c){
      SHIn[n*s.C*sH*sW + c*sH*sW + h*sW + w]
        = (h >= s.pad
            && h < s.H+s.pad
            && w >= s.pad
            && w < s.W+s.pad)
        ? In[n*s.C*s.H*s.W + c*s.H*s.W + (h-s.pad)*s.W + (w-s.pad)]
        : 0.0f;

    }
  }

  for (int x = 0; x < s.X; ++x)
  for (int r = 0; r < s.Rank; ++r)
  for (int n = 0; n < s.N; ++n)
  for (int h = 0; h < s.H; ++h)
  for (int w = 0; w < s.W; ++w)
  for (int t = 0; t < s.T; ++t)
  for (int c = 0; c < s.C; ++c)
  for (int y = 0; y < s.Y; ++y)
    FX[x*s.Rank + r] += SHOut[n*s.T*s.H*s.W + t*s.H*s.W + h*s.W +w]
      *SHIn[n*s.C*s.H*s.W + c*s.H*s.W + (h+y)*s.W + (w+x)]
      *FT[t*s.Rank + r]
      *FC[c*s.Rank + r]
      *FY[y*s.Rank + r];

  return 0;
}
