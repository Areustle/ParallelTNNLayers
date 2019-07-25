#include <cudnn.h>
#include <iostream>

int main(){

  size_t PROFCOUNT = 1;

  float* U;
  float* V;

  int N = 1;
  int C = 16;
  int H = 32;
  int W = 32;
  int fK = 16;
  int fH = 3;
  int fW = 3;

  cudaMalloc(&U, (N*C*H*W)*sizeof(float));
  cudaMalloc(&V, (N*fK*H*W)*sizeof(float));

  /* Begin cuDNN Full Convolution profile section */

  cudnnHandle_t cudnn;
  cudnnCreate(&cudnn);

  cudnnTensorDescriptor_t input_descriptor;
  cudnnCreateTensorDescriptor(&input_descriptor);
  cudnnSetTensor4dDescriptor(input_descriptor, CUDNN_TENSOR_NCHW,
                               CUDNN_DATA_FLOAT, N, C, H, W);
  cudnnTensorDescriptor_t output_descriptor;
  cudnnCreateTensorDescriptor(&output_descriptor);
  cudnnSetTensor4dDescriptor(output_descriptor, CUDNN_TENSOR_NCHW,
                               CUDNN_DATA_FLOAT, N, fK, H, W);
  cudnnFilterDescriptor_t kernel_descriptor;
  cudnnCreateFilterDescriptor(&kernel_descriptor);
  cudnnSetFilter4dDescriptor(kernel_descriptor, CUDNN_DATA_FLOAT,
                               CUDNN_TENSOR_NCHW, fK, C, fH, fW);
  cudnnConvolutionDescriptor_t convolution_descriptor;
  cudnnCreateConvolutionDescriptor(&convolution_descriptor);
  cudnnSetConvolution2dDescriptor(convolution_descriptor, 1, 1, 1, 1, 1, 1,
                               CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);

  cudnnConvolutionFwdAlgo_t convolution_algorithm;
  cudnnGetConvolutionForwardAlgorithm(cudnn,
                                      input_descriptor,
                                      kernel_descriptor,
                                      convolution_descriptor,
                                      output_descriptor,
                                      CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                      /*memoryLimitInBytes=*/0,
                                      &convolution_algorithm);
  size_t workspace_bytes = 0;
  cudnnGetConvolutionForwardWorkspaceSize(cudnn,
                                           input_descriptor,
                                           kernel_descriptor,
                                           convolution_descriptor,
                                           output_descriptor,
                                           convolution_algorithm,
                                           &workspace_bytes);

  void* d_workspace;
  cudaMalloc(&d_workspace, workspace_bytes);

  void * K;
  cudaMalloc(&K, (fK*C*fH*fW)*sizeof(float));

  const float alpha = 1, beta = 0;
  for (int i = 0; i<PROFCOUNT; ++i){
    cudnnConvolutionForward(cudnn, &alpha,
        input_descriptor, U,
        kernel_descriptor, K,
        convolution_descriptor, convolution_algorithm,
        d_workspace, workspace_bytes, &beta,
        output_descriptor, V);
    cudaDeviceSynchronize();
  }

  cudaFree(U);
  cudaFree(V);
  cudaFree(K);
  cudaFree(d_workspace);

  cudnnDestroyTensorDescriptor(input_descriptor);
  cudnnDestroyTensorDescriptor(output_descriptor);
  cudnnDestroyFilterDescriptor(kernel_descriptor);
  cudnnDestroyConvolutionDescriptor(convolution_descriptor);
  cudnnDestroy(cudnn);

}
