#include "Utils.cuh"
#include "cp4Conv2d.cuh"
#include "cp4Conv2dBackwardData.cuh"
#include "cp4Conv2dBackwardFilter.cuh"
#include "cp4Conv2dForward.cuh"
#include <iostream>
#include <stdlib.h>

using namespace std;

/*******************************************************************************
 * Unified memory Tensorized call of Convolution in GPU
 ******************************************************************************/
Tensor CP::Conv2dForward(Tensor const Input,
                         Tensor const FT,
                         Tensor const FC,
                         Tensor const FY,
                         Tensor const FX,
                         unsigned     pad) {

  tensor_shape params;
  params.N    = Input.shape[0];
  params.C    = Input.shape[1];
  params.H    = Input.shape[2];
  params.W    = Input.shape[3];
  params.pad  = pad;
  params.Rank = FT.shape[1];
  params.T    = FT.shape[0];
  params.Y    = FY.shape[0];
  params.X    = FX.shape[0];

  Tensor Out{ params.N, params.T, params.H, params.W };

  cp4_conv2d_forward_gpu(params,
                         Input.m_data,
                         FT.m_data,
                         FC.m_data,
                         FY.m_data,
                         FX.m_data,
                         Out.m_data);

  return Out;
}

/*******************************************************************************
 * Unified memory Tensorized call of Convolution Backward Data in GPU
 ******************************************************************************/
Tensor CP::Conv2dBackwardData(Tensor const Upstream,
                              Tensor const FT,
                              Tensor const FC,
                              Tensor const FY,
                              Tensor const FX,
                              unsigned     pad) {

  tensor_shape params;
  params.N    = Upstream.shape[0];
  params.T    = Upstream.shape[1];
  params.H    = Upstream.shape[2];
  params.W    = Upstream.shape[3];
  params.pad  = pad;
  params.Rank = FT.shape[1];
  params.C    = FC.shape[0];
  params.Y    = FY.shape[0];
  params.X    = FX.shape[0];

  Tensor Out{ params.N, params.C, params.H, params.W };

  cp4_conv2d_backward_data_gpu(params,
                               Upstream.m_data,
                               FT.m_data,
                               FC.m_data,
                               FY.m_data,
                               FX.m_data,
                               Out.m_data);

  return Out;
}


/*******************************************************************************
 * Unified memory Tensorized call of Convolution Backward Filter in GPU
 ******************************************************************************/
Tensor CP::Conv2dBackwardFilter(Tensor const dLdO,
                                Tensor const In,
                                Tensor const FT,
                                Tensor const FC,
                                Tensor const FY,
                                Tensor const FX,
                                unsigned     pad) {

  tensor_shape s;
  s.N    = dLdO.shape[0];
  s.T    = dLdO.shape[1];
  s.H    = dLdO.shape[2];
  s.W    = dLdO.shape[3];
  s.pad  = pad;
  s.Rank = FT.shape[1];
  s.C    = FC.shape[0];
  s.Y    = FY.shape[0];
  s.X    = FX.shape[0];

  Tensor dFT{ s.T, s.Rank };
  Tensor dFC{ s.C, s.Rank };
  Tensor dFY{ s.Y, s.Rank };
  Tensor dFX{ s.X, s.Rank };
  /* Tensor dFF{ s.T, s.C, s.Y, s.X }; */


  /* cp4_conv2d_backward_filter_full_gpu( */
  /*     s, dFF.m_data, In.m_data, dLdO.m_data); */

  /* return dFF; */

  cp4_conv2d_backward_filter_t_gpu(s,
                                   dFT.m_data,
                                   In.m_data,
                                   dLdO.m_data,
                                   FC.m_data,
                                   FY.m_data,
                                   FX.m_data);

  /* cout << dFT.m_data[0] << endl; */
  /* cout << dFC.m_data[0] << endl; */
  /* cout << dFY.m_data[0] << endl; */
  /* cout << dFX.m_data[0] << endl; */

  cp4_conv2d_backward_filter_c_gpu(s,
                                   dFC.m_data,
                                   In.m_data,
                                   dLdO.m_data,
                                   FT.m_data,
                                   FY.m_data,
                                   FX.m_data);

  cp4_conv2d_backward_filter_y_gpu(s,
                                   dFY.m_data,
                                   In.m_data,
                                   dLdO.m_data,
                                   FT.m_data,
                                   FC.m_data,
                                   FX.m_data);


  cp4_conv2d_backward_filter_x_gpu(s,
                                   dFX.m_data,
                                   In.m_data,
                                   dLdO.m_data,
                                   FT.m_data,
                                   FC.m_data,
                                   FY.m_data);

  return cp4recom(dFT, dFC, dFY, dFX);
}

/*******************************************************************************
 * Run_convolution operation with a profile count loop
 ******************************************************************************/
float CP::run_convolution(tensor_shape p, unsigned PROFCOUNT) {

  float* In;
  float* Out;
  float* FT;
  float* FC;
  float* FX;
  float* FY;

  cudaMalloc(&In, p.N * p.C * p.H * p.W * sizeof(float));
  cudaMalloc(&FT, p.T * p.Rank * sizeof(float));
  cudaMalloc(&FC, p.C * p.Rank * sizeof(float));
  cudaMalloc(&FY, p.Y * p.Rank * sizeof(float));
  cudaMalloc(&FX, p.X * p.Rank * sizeof(float));
  cudaMalloc(&Out, p.N * p.T * p.H * p.W * sizeof(float));


  float us = cp4_conv2d_forward_gpu(p, In, FT, FC, FY, FX, Out, PROFCOUNT);

  cudaFree(In);
  cudaFree(FT);
  cudaFree(FC);
  cudaFree(FY);
  cudaFree(FX);
  cudaFree(Out);

  return us;
}


/*******************************************************************************
 * Main function. call 1 instance of kernel execution
 ******************************************************************************/
int main(int argc, char** argv) {

  unsigned N    = 5;
  unsigned C    = 32;
  unsigned H    = 1024;
  unsigned W    = 1024;
  unsigned pad  = 1;
  unsigned T    = 32;
  unsigned Y    = 3;
  unsigned X    = 3;
  unsigned Rank = 8;

  if (argc != 11) {
    cerr << "Using Default shape" << endl;
    cudaSetDevice(0);
  } else {
    N    = atoi(argv[1]);
    C    = atoi(argv[2]);
    H    = atoi(argv[3]);
    W    = atoi(argv[4]);
    pad  = atoi(argv[5]);
    T    = atoi(argv[6]);
    Y    = atoi(argv[7]);
    X    = atoi(argv[8]);
    Rank = atoi(argv[9]);
    cudaSetDevice(atoi(argv[10]));
  }

  tensor_shape params;
  params.N    = N;
  params.C    = C;
  params.H    = H;
  params.W    = W;
  params.pad  = pad;
  params.Rank = Rank;
  params.T    = T;
  params.Y    = Y;
  params.X    = X;

  CP::run_convolution(params, 1);
}
