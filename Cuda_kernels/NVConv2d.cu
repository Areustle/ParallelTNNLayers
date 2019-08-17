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

float conv2d_forward_gpu(tensor_shape params,
                         float*       In,
                         float*       Filter,
                         float*       Out,
                         unsigned     PROFCOUNT = 1) {

  const unsigned N   = params.N;
  const unsigned C   = params.C;
  const unsigned H   = params.H;
  const unsigned W   = params.W;
  const unsigned pad = params.pad;
  const unsigned fK  = params.fK;
  const unsigned fH  = params.fH;
  const unsigned fW  = params.fW;

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
  cudaMalloc(&d_workspace, workspace_bytes);

  const float alpha = 1, beta = 0;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  float cumulativeNS = 0.0f;
  for (unsigned i = 0; i < PROFCOUNT; ++i) {
    cudaDeviceSynchronize();
    cudaEventRecord(start);
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
    cudaDeviceSynchronize();
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cumulativeNS += milliseconds * 1e6;
  }
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  cudaFree(d_workspace);
  cudnnDestroyTensorDescriptor(input_descriptor);
  cudnnDestroyTensorDescriptor(output_descriptor);
  cudnnDestroyFilterDescriptor(kernel_descriptor);
  cudnnDestroyConvolutionDescriptor(convolution_descriptor);
  cudnnDestroy(cudnn);

  return (cumulativeNS / PROFCOUNT);
}


/*******************************************************************************
 * Unified memory Tensorized call of Convolution
 ******************************************************************************/
Tensor NV::Conv2dForward(const Tensor In, const Tensor K, unsigned pad) {

  tensor_shape params;
  params.N   = In.shape[0];
  params.C   = In.shape[1];
  params.H   = In.shape[2];
  params.W   = In.shape[3];
  params.pad = pad;
  params.fK  = K.shape[0];
  params.fH  = K.shape[2];
  params.fW  = K.shape[3];

  Tensor V({ In.shape[0], K.shape[0], In.shape[2], In.shape[3] });
  conv2d_forward_gpu(params, In.m_data, K.m_data, V.m_data, 1);

  return V;
}


/*******************************************************************************
 * run_convolution operation with a profile count loop
 ******************************************************************************/
float NV::run_convolution(tensor_shape p, unsigned PROFCOUNT) {

  float* In;
  float* Out;
  float* Filter;


  cudaMalloc(&In, p.N * p.C * p.H * p.W * sizeof(float));
  cudaMalloc(&Filter, p.fK * p.C * p.fH * p.fW * sizeof(float));
  cudaMalloc(&Out, p.N * p.fK * p.H * p.W * sizeof(float));

  float ns = conv2d_forward_gpu(p, In, Filter, Out, PROFCOUNT);

  cudaFree(In);
  cudaFree(Filter);
  cudaFree(Out);

  return ns;
}


/*******************************************************************************
 * Main function. call 1 instance of kernel execution
 ******************************************************************************/
int main(int argc, char** argv) {

  unsigned N   = 5;
  unsigned C   = 32;
  unsigned H   = 1024;
  unsigned W   = 1024;
  unsigned pad = 1;
  unsigned fK  = 32;
  unsigned fH  = 3;
  unsigned fW  = 3;

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
  }

  tensor_shape params;
  params.N   = N;
  params.C   = C;
  params.H   = H;
  params.W   = W;
  params.pad = pad;
  params.fK  = fK;
  params.fH  = fH;
  params.fW  = fW;

  NV::run_convolution(params, 1);
}
