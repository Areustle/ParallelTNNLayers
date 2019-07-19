#include "cudnnConv2d.cuh"
#include <cudnn.h>
#include <iostream>

#define checkCUDNN(expression)                               \
  {                                                          \
    cudnnStatus_t status = (expression);                     \
    if (status != CUDNN_STATUS_SUCCESS) {                    \
      std::cerr << "Error on line " << __LINE__ << ": "      \
                << cudnnGetErrorString(status) << std::endl; \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
  }

Tensor nn_conv2d(Tensor const U, Tensor const K) {

  cudnnHandle_t cudnn;
  cudnnCreate(&cudnn);

  cudnnTensorDescriptor_t input_descriptor;
  checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
  checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor,
                                        CUDNN_TENSOR_NCHW,
                                        CUDNN_DATA_FLOAT,
                                        U.shape[0],
                                        U.shape[1],
                                        U.shape[2],
                                        U.shape[3]));

  cudnnFilterDescriptor_t kernel_descriptor;
  checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor));
  checkCUDNN(cudnnSetFilter4dDescriptor(kernel_descriptor,
                                        CUDNN_DATA_FLOAT,
                                        CUDNN_TENSOR_NCHW,
                                        K.shape[0],
                                        K.shape[1],
                                        K.shape[2],
                                        K.shape[3]));

  cudnnConvolutionDescriptor_t convolution_descriptor;
  checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
  checkCUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor,
                                             /*pad_height=*/1,
                                             /*pad_width=*/1,
                                             /*vertical_stride=*/1,
                                             /*horizontal_stride=*/1,
                                             /*dilation_height=*/1,
                                             /*dilation_width=*/1,
                                             /*mode=*/CUDNN_CROSS_CORRELATION,
                                             /*computeType=*/CUDNN_DATA_FLOAT));

  int batch_size{ 0 }, channels{ 0 }, height{ 0 }, width{ 0 };
  checkCUDNN(cudnnGetConvolution2dForwardOutputDim(convolution_descriptor,
                                                   input_descriptor,
                                                   kernel_descriptor,
                                                   &batch_size,
                                                   &channels,
                                                   &height,
                                                   &width));

  cudnnTensorDescriptor_t output_descriptor;
  checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
  checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor,
                                        CUDNN_TENSOR_NCHW,
                                        CUDNN_DATA_FLOAT,
                                        batch_size,
                                        channels,
                                        height,
                                        width));


  cudnnConvolutionFwdAlgo_t convolution_algorithm;
  checkCUDNN(
      cudnnGetConvolutionForwardAlgorithm(cudnn,
                                          input_descriptor,
                                          kernel_descriptor,
                                          convolution_descriptor,
                                          output_descriptor,
                                          CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                          /*memoryLimitInBytes=*/0,
                                          &convolution_algorithm));

  size_t workspace_bytes{ 0 };
  checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
                                                     input_descriptor,
                                                     kernel_descriptor,
                                                     convolution_descriptor,
                                                     output_descriptor,
                                                     convolution_algorithm,
                                                     &workspace_bytes));

  void* d_workspace{ nullptr };
  cudaMallocManaged(&d_workspace, workspace_bytes);

  float* d_input{ nullptr };
  cudaMalloc(&d_input, U.size());
  cudaMemcpy(d_input, U.m_data, U.size(), cudaMemcpyHostToDevice);

  size_t out_bytes = batch_size * channels * height * width * sizeof(float);
  float* d_output{ nullptr };
  cudaMalloc(&d_output, out_bytes);
  cudaMemset(d_output, 0, out_bytes);

  Tensor V({ batch_size, channels, height, width });

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

  cudaFree(d_workspace);
  cudnnDestroyTensorDescriptor(input_descriptor);
  cudnnDestroyTensorDescriptor(output_descriptor);
  cudnnDestroyFilterDescriptor(kernel_descriptor);
  cudnnDestroyConvolutionDescriptor(convolution_descriptor);
  cudnnDestroy(cudnn);

  return V;
}
