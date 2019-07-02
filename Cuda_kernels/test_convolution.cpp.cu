/* #include "CudaAllocator.h" */
/* #include "Tensor.h" */
#include <iostream>
/* #include <random> */

using namespace std;

int main() {

  /* const size_t  dN = 1, dC = 1, dH = 1, dW = 4; //, dF = 16, dKH = 3, dKW = 3; */
  /* random_device rd; */
  /* mt19937       gen(rd()); */
  /* uniform_real_distribution<> dis(-1.0, 1.0); */

  /* auto random_fill = [&dis, &gen](size_t len, float* A) { */
  /*   for (size_t i = 0; i < len; ++i) */
  /*     A[i] = dis(gen); */
  /* }; */

  /* size_t len_input = (dN * dC * dH * dW); */

  /* Tensor input(dN, dC, dH, dW); */
  /* random_fill(len_input, input.data); */

  /* Tensor output(input); */

  /* if (input.data[0] == output.data[0]) */
  /*   cout << "EQUAL" << endl; */
  /* else */
  /*   cout << "NOT EQUAL" << endl; */

  /* CHECK(input.data[0] == doctest::Approx(output.data[0]).epsilon(1e-3)); */

  /* for (int i = 0; i < input.len; ++i) { */
  /*   CHECK(input.data[i] == doctest::Approx(output.data[i]).epsilon(1e-3)); */
  /* } */

  float* input;
  float* output;
  cudaMallocManaged(&input, 4*sizeof(float));
  cudaMemset(input, 4, 4);
  /* input[0] = 7; */
  cudaMallocManaged(&output, 4*sizeof(float));
  cudaMemcpy(output, input, 4, cudaMemcpyDeviceToDevice);

  cout << input[0] << "____" << output[0] << endl;

  cudaFree(input);
  cudaFree(output);
}
