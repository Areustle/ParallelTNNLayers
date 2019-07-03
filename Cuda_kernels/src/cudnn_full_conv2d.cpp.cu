#include "Cuda_kernels/cudnn_full_conv2d.h"
#include <cudnn.h>
#include <iostream>


CudnnConv2d::CudnnConv2d(const size_t N,
                         const size_t C,
                         const size_t H,
                         const size_t W,
                         const size_t F,
                         const size_t Y,
                         const size_t X)
    : N(N)
    , C(C)
    , H(H)
    , W(W)
    , F(F)
    , Y(Y)
    , X(X) {

  cudnnCreate(&cudnn);
  cudnnCreateTensorDescriptor(&input_descriptor);
  cudnnSetTensor4dDescriptor(
      input_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C, H, W);
  cudnnCreateTensorDescriptor(&output_descriptor);
  cudnnSetTensor4dDescriptor(
      output_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C, H, W);
  cudnnCreateFilterDescriptor(&kernel_descriptor);
  cudnnSetFilter4dDescriptor(
      kernel_descriptor, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, C, F, Y, X);
  cudnnCreateConvolutionDescriptor(&convolution_descriptor);
  cudnnSetConvolution2dDescriptor(convolution_descriptor,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  1,
                                  CUDNN_CROSS_CORRELATION,
                                  CUDNN_DATA_FLOAT);
  cudnnGetConvolutionForwardAlgorithm(cudnn,
                                      input_descriptor,
                                      kernel_descriptor,
                                      convolution_descriptor,
                                      output_descriptor,
                                      CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                      /*memoryLimitInBytes=*/0,
                                      &convolution_algorithm);
  cudnnGetConvolutionForwardWorkspaceSize(cudnn,
                                          input_descriptor,
                                          kernel_descriptor,
                                          convolution_descriptor,
                                          output_descriptor,
                                          convolution_algorithm,
                                          &workspace_bytes);

  cudaMalloc(&d_workspace, workspace_bytes);
}

CudnnConv2d::~CudnnConv2d() {
  cudaFree(d_workspace);
  cudnnDestroyTensorDescriptor(input_descriptor);
  cudnnDestroyTensorDescriptor(output_descriptor);
  cudnnDestroyFilterDescriptor(kernel_descriptor);
  cudnnDestroyConvolutionDescriptor(convolution_descriptor);
  cudnnDestroy(cudnn);
}


void CudnnConv2d::opearator()(float* Input, float* Kernel, float* Output) {
  const float alpha = 1, beta = 0;
  cudnnConvolutionForward(cudnn,
                          &alpha,
                          input_descriptor,
                          Input,
                          kernel_descriptor,
                          Kernel,
                          convolution_descriptor,
                          convolution_algorithm,
                          d_workspace,
                          workspace_bytes,
                          &beta,
                          output_descriptor,
                          Output);
}

