#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;  // NOLINT(build/namespaces)

REGISTER_OP("CpForwardUnfused")
    .Input("U: float")
    .Input("K0: float")
    .Output("U0: float")
    .Doc(R"doc(Calculate the CP decomposed forward convolution operator)doc");

void CpForwardUnfusedKernelLauncher(const float* U, const float* K0, float* V);

class CpForwardUnfusedOp : public OpKernel {
 public:
  explicit CpForwardUnfusedOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensors
    const Tensor& tenU = context->input(0);
    const Tensor& tenK0 = context->input(1);
    OP_REQUIRES(context, tenU.shape().dims()==4,
        errors::InvalidArgument("CpForwardUnfusedOp expects Input U to be a rank 4 Tensor NHWC"));
    OP_REQUIRES(context, tenU.shape().dims_size(0)==8,
        errors::InvalidArgument("U[0] != 8"));
    OP_REQUIRES(context, tenU.shape().dims_size(1)==32,
        errors::InvalidArgument("U[1] != 32"));
    OP_REQUIRES(context, tenU.shape().dims_size(2)==32,
        errors::InvalidArgument("U[2] != 32"));
    OP_REQUIRES(context, tenU.shape().dims_size(3)==16,
        errors::InvalidArgument("U[3] != 16"));
    OP_REQUIRES(context, tenK0.shape().dims_size(0)==16,
        errors::InvalidArgument("K0[0] != 16"));
    OP_REQUIRES(context, tenK0.shape().dims_size(0)==6,
        errors::InvalidArgument("K0[1] != 6"));
    OP_REQUIRES(context, TensorShapeUtils::IsMatrix(tenK0.shape()),
        errors::InvalidArgument("CpForwardUnfusedOp expects Input B to be a Matrix (rank 2 Tensor)"));
    OP_REQUIRES(context, tenU.shape().dim_size(3) == tenK0.shape().dim_size(0),
        errors::InvalidArgument("CpForwardUnfusedOp expects Input A, Mode 2 to be equal length with Input B Mode 0"));

    // Create an output tensor
    Tensor* output_tensor = nullptr;
    auto outshape = tenU.shape();
    outshape.set_dim(3, tenK0.shape().dim_size(1));
    OP_REQUIRES_OK(context, context->allocate_output(0, outshape,
                                                     &output_tensor));

    auto U = tenU.flat<float>();
    auto K0 = tenK0.flat<float>();
    auto U0 = output_tensor->template flat<float>();

    // Call the cuda kernel launcher
    CpForwardUnfusedKernelLauncher(U.data(), K0.data(), U0.data());
  }
};

REGISTER_KERNEL_BUILDER(Name("CpForwardUnfused").Device(DEVICE_GPU), CpForwardUnfusedOp);
