#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;  // NOLINT(build/namespaces)

REGISTER_OP("CpForwardUnfused")
    .Input("input: float")
    .Input("kernel0: float")
    .Output("outputu0: float")
    .Doc(R"doc(Calculate the first step of CP decomposed forward convolution operator)doc");

/* void CpForwardUnfusedKernelLauncher(const float* input, const float* kernel0, float* V); */
void Cp0NhwcKernelLauncher(const float* U, const float* K0 float* U0);

class Cp0NhwcOp : public OpKernel {
 public:
  explicit Cp0NhwcOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensors
    const Tensor& tenU = context->input(0);
    const Tensor& tenK0 = context->input(1);
    const Tensor& tenK1 = context->input(2);
    const Tensor& tenK2 = context->input(3);
    OP_REQUIRES(context, tenU.shape().dims()==4,
        errors::InvalidArgument("Cp0NhwcOp expects Input image to be a rank 4 Tensor NHWC"));
    OP_REQUIRES(context, tenK0.shape().dims()==2,
        errors::InvalidArgument("Cp0NhwcOp expects Kernel 0 to be a rank 2 Tensor [KchIN, KchOUT]"));

    OP_REQUIRES(context, tenU.shape().dim_size(0)==8,
        errors::InvalidArgument("input[0] != 8"));
    OP_REQUIRES(context, tenU.shape().dim_size(1)==32,
        errors::InvalidArgument("input[1] != 32"));
    OP_REQUIRES(context, tenU.shape().dim_size(2)==32,
        errors::InvalidArgument("input[2] != 32"));
    OP_REQUIRES(context, tenU.shape().dim_size(3)==16,
        errors::InvalidArgument("input[3] != 16"));

    OP_REQUIRES(context, tenK0.shape().dim_size(0)==16,
        errors::InvalidArgument("kernel0[0] != 16"));
    OP_REQUIRES(context, tenK0.shape().dim_size(1)==6,
        errors::InvalidArgument("kernel0[1] != 6"));

    // Create u0 output tensor
    Tensor* u0_tensor = nullptr;
    auto u0_outshape = tenU.shape();
    u0_outshape.set_dim(3, tenK0.shape().dim_size(1));
    OP_REQUIRES_OK(context, context->allocate_output(0, u0_outshape,
                                                     &u0_tensor));

    auto input = tenU.flat<float>();
    auto kernel0 = tenK0.flat<float>();
    auto outputu0 = u0_tensor->template flat<float>();

    // Call the cuda kernel0 launcher
    Cp0NhwcKernelLauncher(input.data(), kernel0.data(), outputu0.data());
  }
};

REGISTER_KERNEL_BUILDER(Name("Cp0Nhwc").Device(DEVICE_GPU), Cp0NhwcOp);
