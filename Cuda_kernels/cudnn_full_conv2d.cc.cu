#include "cudnn_full_conv2d.h"
#include <cudnn.h>
#include <iostream>

void cudnn_imp::conv2d( float*       U,
                        void*        K,
                        float*       V,
                        const size_t dN,
                        const size_t dC,
                        const size_t dH,
                        const size_t dW,
                        const size_t dF,
                        const size_t dKH,
                        const size_t dKW ) {

  void* d_workspace;

  cudnnHandle_t                cudnn;
  cudnnTensorDescriptor_t      input_descriptor;
  cudnnTensorDescriptor_t      output_descriptor;
  cudnnFilterDescriptor_t      kernel_descriptor;
  cudnnConvolutionDescriptor_t convolution_descriptor;
  cudnnConvolutionFwdAlgo_t    convolution_algorithm;
  size_t                       workspace_bytes = 0;

  cudnnCreate( &cudnn );
  cudnnCreateTensorDescriptor( &input_descriptor );
  cudnnSetTensor4dDescriptor(
      input_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, dN, dC, dH, dW );
  cudnnCreateTensorDescriptor( &output_descriptor );
  cudnnSetTensor4dDescriptor(
      output_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, dN, dC, dH, dW );
  cudnnCreateFilterDescriptor( &kernel_descriptor );
  cudnnSetFilter4dDescriptor( kernel_descriptor,
                              CUDNN_DATA_FLOAT,
                              CUDNN_TENSOR_NCHW,
                              dC,
                              dF,
                              dKH,
                              dKW );
  cudnnCreateConvolutionDescriptor( &convolution_descriptor );
  cudnnSetConvolution2dDescriptor( convolution_descriptor,
                                   1,
                                   1,
                                   1,
                                   1,
                                   1,
                                   1,
                                   CUDNN_CROSS_CORRELATION,
                                   CUDNN_DATA_FLOAT );
  cudnnGetConvolutionForwardAlgorithm( cudnn,
                                       input_descriptor,
                                       kernel_descriptor,
                                       convolution_descriptor,
                                       output_descriptor,
                                       CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                       /*memoryLimitInBytes=*/0,
                                       &convolution_algorithm );
  cudnnGetConvolutionForwardWorkspaceSize( cudnn,
                                           input_descriptor,
                                           kernel_descriptor,
                                           convolution_descriptor,
                                           output_descriptor,
                                           convolution_algorithm,
                                           &workspace_bytes );
  /* cerr << "Workspace size: " << workspace_bytes << endl; */


  /* cudaMalloc( &U, ( dN * dC * dH * dW ) * sizeof( float ) ); */
  /* cudaMalloc( &V, ( dN * dC * dH * dW ) * sizeof( float ) ); */
  /* cudaMalloc( &K, ( dC * dF * dKH * dKW ) * sizeof( float ) ); */

  cudaMalloc( &d_workspace, workspace_bytes );

  const float alpha = 1, beta = 0;
  cudnnConvolutionForward( cudnn,
                           &alpha,
                           input_descriptor,
                           U,
                           kernel_descriptor,
                           K,
                           convolution_descriptor,
                           convolution_algorithm,
                           d_workspace,
                           workspace_bytes,
                           &beta,
                           output_descriptor,
                           V );
  cudaFree( d_workspace );

  /* cudaFree( U ); */
  /* cudaFree( V ); */
  /* cudaFree( K ); */

  cudnnDestroyTensorDescriptor( input_descriptor );
  cudnnDestroyTensorDescriptor( output_descriptor );
  cudnnDestroyFilterDescriptor( kernel_descriptor );
  cudnnDestroyConvolutionDescriptor( convolution_descriptor );
  cudnnDestroy( cudnn );
}
