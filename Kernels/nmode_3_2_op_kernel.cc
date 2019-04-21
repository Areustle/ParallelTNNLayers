#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;  // NOLINT(build/namespaces)

REGISTER_OP("NMode32")
    .Input("a: float")
    .Input("b: float")
    .Output("output: float")
    .Doc(R"doc(
Calculate the n-mode product C = A x_2 B where A is a 3rd order Tensor and
B is a 2nd order Tensor.

output: A 3rd order Tensor.
  C[i,j,r] += A[i,j,s] * B[s,r]
)doc");

void NMode32KernelLauncher(const float* A, const int I, const int J, const int S,
                   const float* B, const int R,
                   float* C);

class NMode32Op : public OpKernel {
 public:
  explicit NMode32Op(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensors
    const Tensor& tenA = context->input(0);
    const Tensor& tenB = context->input(1);
    OP_REQUIRES(context, tenA.shape().dims()==3,
        errors::InvalidArgument("NMode32 expects Input A to be a rank 3 Tensor"));
    OP_REQUIRES(context, TensorShapeUtils::IsMatrix(tenB.shape()),
        errors::InvalidArgument("NMode32 expects Input B to be a Matrix (rank 2 Tensor)"));
    OP_REQUIRES(context, tenA.shape().dim_size(2) == tenB.shape().dim_size(0),
        errors::InvalidArgument("NMode32 expects Input A, Mode 2 to be equal length with Input B Mode 0"));

    // Create an output tensor
    Tensor* output_tensor = nullptr;
    auto outshape = tenA.shape();
    outshape.set_dim(2, tenB.shape().dim_size(1));
    OP_REQUIRES_OK(context, context->allocate_output(0, outshape,
                                                     &output_tensor));

    auto A = tenA.flat<float>();
    auto B = tenB.flat<float>();
    auto C = output_tensor->template flat<float>();

    const int I = tenA.shape().dim_size(0);
    const int J = tenA.shape().dim_size(1);
    const int S = tenA.shape().dim_size(2);
    const int R = tenB.shape().dim_size(0);

    // Call the cuda kernel launcher
    NMode32KernelLauncher(A.data(), I, J, S, B.data(), R, C.data());
  }
};

REGISTER_KERNEL_BUILDER(Name("NMode32").Device(DEVICE_GPU), NMode32Op);
