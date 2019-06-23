#include <cudnn.h>
#include <iostream>

using namespace std;

int main() {

  size_t PROFCOUNT = 100000;

  float* U;
  void*  K;
  float* V;
  void*  d_workspace;

  const size_t dN = 1, dC = 16, dH = 32, dW = 32, dKN = 16, dKH = 3, dKW = 3;

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
                              dKN,
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
  cerr << "Workspace size: " << workspace_bytes << endl;


  cudaMalloc( &U, ( 1 * 16 * 32 * 32 ) * sizeof( float ) );
  cudaMalloc( &V, ( 1 * 16 * 32 * 32 ) * sizeof( float ) );
  cudaMalloc( &d_workspace, workspace_bytes );
  cudaMalloc( &K, ( 16 * 16 * 3 * 3 ) * sizeof( float ) );


  const float alpha = 1, beta = 0;
  for ( int i = 0; i < PROFCOUNT; ++i )
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

  cudaFree( U );
  cudaFree( V );
  cudaFree( K );
  cudaFree( d_workspace );

  cudnnDestroyTensorDescriptor( input_descriptor );
  cudnnDestroyTensorDescriptor( output_descriptor );
  cudnnDestroyFilterDescriptor( kernel_descriptor );
  cudnnDestroyConvolutionDescriptor( convolution_descriptor );
  cudnnDestroy( cudnn );
}
