#include "cudnn_full_conv2d.h"
#include <cudnn.h>

Tensor nn_conv2d(Tensor const U, Tensor const K) {

  void*                        d_workspace;
  cudnnHandle_t                cudnn;
  cudnnTensorDescriptor_t      input_descriptor;
  cudnnTensorDescriptor_t      output_descriptor;
  cudnnFilterDescriptor_t      kernel_descriptor;
  cudnnConvolutionDescriptor_t convolution_descriptor;
  cudnnConvolutionFwdAlgo_t    convolution_algorithm;
  size_t                       workspace_bytes = 0;

  cudnnCreate(&cudnn);
  cudnnCreateTensorDescriptor(&input_descriptor);
  cudnnSetTensor4dDescriptor(
      input_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, U.N, U.C, U.H, U.W);
  cudnnCreateTensorDescriptor(&output_descriptor);
  cudnnSetTensor4dDescriptor(
      output_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, U.N, U.C, U.H, U.W);
  cudnnCreateFilterDescriptor(&kernel_descriptor);
  cudnnSetFilter4dDescriptor(
      kernel_descriptor, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, K.N, K.C, K.H, K.W);
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

  cudaMallocManaged(&d_workspace, workspace_bytes);

  Tensor V(U);

  const float alpha = 1, beta = 0;
  cudnnConvolutionForward(cudnn,
                          &alpha,
                          input_descriptor,
                          U.m_data,
                          kernel_descriptor,
                          K.m_data,
                          convolution_descriptor,
                          convolution_algorithm,
                          d_workspace,
                          workspace_bytes,
                          &beta,
                          output_descriptor,
                          V.m_data);

  cudaDeviceSynchronize();
  cudaFree(d_workspace);
  cudnnDestroyTensorDescriptor(input_descriptor);
  cudnnDestroyTensorDescriptor(output_descriptor);
  cudnnDestroyFilterDescriptor(kernel_descriptor);
  cudnnDestroyConvolutionDescriptor(convolution_descriptor);
  cudnnDestroy(cudnn);

  return V;
}


#define DOCTEST_CONFIG_IMPLEMENTATION_IN_DLL
#include "doctest.h"

TEST_CASE("cudnn_full_conv2d test") {
  Tensor U(1, 1, 32, 32);
  Tensor K(1, 1, 3, 3);

  auto V = nn_conv2d(U,K);

  for (int i = 0; i < V.size(); ++i) {
    CHECK(V[i] == 0);
  }
}
