#include "NVConv2d.cuh"
#include <cudnn.h>
#include <iostream>
#include <stdlib.h>

using namespace std;

#define checkCUDNN(expression)                               \
  {                                                          \
    cudnnStatus_t status = (expression);                     \
    if (status != CUDNN_STATUS_SUCCESS) {                    \
      std::cerr << "Error on line " << __LINE__ << ": "      \
                << cudnnGetErrorString(status) << std::endl; \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
  }

void NV::conv2d_forward_gpu(float*   In,
                            unsigned N,
                            unsigned C,
                            unsigned H,
                            unsigned W,
                            unsigned pad,
                            float*   Filter,
                            unsigned fK,
                            unsigned fH,
                            unsigned fW,
                            float*   Out) {
  cudnnHandle_t cudnn;
  cudnnCreate(&cudnn);

  cudnnTensorDescriptor_t input_descriptor;
  checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
  checkCUDNN(cudnnSetTensor4dDescriptor(
      input_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C, H, W));

  cudnnFilterDescriptor_t kernel_descriptor;
  checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor));
  checkCUDNN(cudnnSetFilter4dDescriptor(
      kernel_descriptor, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, fK, C, fH, fW));

  cudnnConvolutionDescriptor_t convolution_descriptor;
  checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
  checkCUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor,
                                             /*pad_height=*/pad,
                                             /*pad_width=*/pad,
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
  cudaMalloc(&d_input, N * C * H * W);
  cudaMemcpy(d_input, In, N * C * H * W, cudaMemcpyHostToDevice);

  size_t out_bytes = batch_size * channels * height * width * sizeof(float);
  float* d_output{ nullptr };
  cudaMalloc(&d_output, out_bytes);
  cudaMemset(d_output, 0, out_bytes);

  const float alpha = 1, beta = 0;
  cudnnConvolutionForward(cudnn,
                          &alpha,
                          input_descriptor,
                          In,
                          kernel_descriptor,
                          Filter,
                          convolution_descriptor,
                          convolution_algorithm,
                          d_workspace,
                          workspace_bytes,
                          &beta,
                          output_descriptor,
                          Out);

  cudaFree(d_workspace);
  cudnnDestroyTensorDescriptor(input_descriptor);
  cudnnDestroyTensorDescriptor(output_descriptor);
  cudnnDestroyFilterDescriptor(kernel_descriptor);
  cudnnDestroyConvolutionDescriptor(convolution_descriptor);
  cudnnDestroy(cudnn);
}

Tensor NV::Conv2dForward(const Tensor In, const Tensor K, unsigned pad) {

  Tensor V({ In.shape[0], K.shape[0], In.shape[2], In.shape[3] });
  NV::conv2d_forward_gpu(In.m_data,
                         In.shape[0],
                         In.shape[1],
                         In.shape[2],
                         In.shape[3],
                         pad,
                         K.m_data,
                         K.shape[0],
                         K.shape[2],
                         K.shape[3],
                         V.m_data);

  return V;
}


int main(int argc, char** argv) {

  unsigned N   = 5;
  unsigned C   = 32;
  unsigned H   = 1024;
  unsigned W   = 1024;
  unsigned pad = 1;
  unsigned fK  = 32;
  unsigned fH  = 3;
  unsigned fW  = 3;

  /* int devcnt = -1; */

  if (argc != 11) {
    cudaSetDevice(0);
    cerr << "Using default shape" << endl;
  } else {
    N   = atoi(argv[1]);
    C   = atoi(argv[2]);
    H   = atoi(argv[3]);
    W   = atoi(argv[4]);
    pad = atoi(argv[5]);
    fK  = atoi(argv[6]);
    fH  = atoi(argv[7]);
    fW  = atoi(argv[8]);
    // fRank var meaningless here
    cudaSetDevice(atoi(argv[10]));
    /* cerr << atoi(argv[10]) << " " << argv[10] << endl; */
  }

  /* cerr << cudaGetDevice(&devcnt) << endl; */
  /* cerr << devcnt << endl; */

  float* In;
  float* Out;
  float* Filter;


  cudaMalloc(&In, N * C * H * W * sizeof(float));
  cudaMalloc(&Filter, fK * C * fH * fW * sizeof(float));
  cudaMalloc(&Out, N * fK * H * W * sizeof(float));

  NV::conv2d_forward_gpu(In, N, C, H, W, pad, Filter, fK, fH, fW, Out);

  cudaFree(In);
  cudaFree(Filter);
  cudaFree(Out);
}
