#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;  // NOLINT(build/namespaces)

REGISTER_OP("CpForwardUnfused")
    .Input("input: float")
    .Input("kernel0: float")
    .Input("kernel1: float")
    .Input("kernel2: float")
    .Output("outputu0: float")
    .Output("outputu1: float")
    .Output("outputv: float")
    .Doc(R"doc(Calculate the CP decomposed forward convolution operator)doc");

/* void CpForwardUnfusedKernelLauncher(const float* input, const float* kernel0, float* V); */
void CpForwardUnfusedKernelLauncher(const float* U,
    const float* K0, const float* K1, const float* K2,
    float* U0, float* U1, float* V);

class CpForwardUnfusedOp : public OpKernel {
 public:
  explicit CpForwardUnfusedOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensors
    const Tensor& tenU = context->input(0);
    const Tensor& tenK0 = context->input(1);
    const Tensor& tenK1 = context->input(2);
    const Tensor& tenK2 = context->input(3);
    OP_REQUIRES(context, tenU.shape().dims()==4,
        errors::InvalidArgument("CpForwardUnfusedOp expects Input image to be a rank 4 Tensor NHWC"));
    OP_REQUIRES(context, tenK0.shape().dims()==2,
        errors::InvalidArgument("CpForwardUnfusedOp expects Kernel 0 to be a rank 2 Tensor [KchIN, KchOUT]"));
    OP_REQUIRES(context, tenK1.shape().dims()==4,
        errors::InvalidArgument("CpForwardUnfusedOp expects Kernel 1 to be a rank 4 Tensor [HWC1]"));
    OP_REQUIRES(context, tenK2.shape().dims()==2,
        errors::InvalidArgument("CpForwardUnfusedOp expects Kernel 2 to be a rank 2 Tensor [KchIN, KchoOUT]"));

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

    OP_REQUIRES(context, tenK1.shape().dim_size(0)==3,
        errors::InvalidArgument("kernel1[0] != 6"));
    OP_REQUIRES(context, tenK1.shape().dim_size(1)==3,
        errors::InvalidArgument("kernel1[1] != 3"));
    OP_REQUIRES(context, tenK1.shape().dim_size(2)==6,
        errors::InvalidArgument("kernel1[2] != 3"));
    OP_REQUIRES(context, tenK1.shape().dim_size(3)==1,
        errors::InvalidArgument("kernel1[3] != 1"));

    OP_REQUIRES(context, tenK2.shape().dim_size(0)==6,
        errors::InvalidArgument("kernel2[0] != 6"));
    OP_REQUIRES(context, tenK2.shape().dim_size(1)==16,
        errors::InvalidArgument("kernel2[1] != 16"));


    // Create u0 output tensor
    Tensor* u0_tensor = nullptr;
    auto u0_outshape = tenU.shape();
    u0_outshape.set_dim(3, tenK0.shape().dim_size(1));
    OP_REQUIRES_OK(context, context->allocate_output(0, u0_outshape,
                                                     &u0_tensor));

    Tensor* u1_tensor = nullptr;
    auto u1_outshape = u0_outshape;
    OP_REQUIRES_OK(context, context->allocate_output(1, u1_outshape,
                                                     &u1_tensor));

    Tensor* v_tensor = nullptr;
    auto v_outshape = tenU.shape();
    OP_REQUIRES_OK(context, context->allocate_output(2, v_outshape,
                                                     &v_tensor));

    auto input = tenU.flat<float>();
    auto kernel0 = tenK0.flat<float>();
    auto kernel1 = tenK1.flat<float>();
    auto kernel2 = tenK2.flat<float>();
    auto outputu0 = u0_tensor->template flat<float>();
    auto outputu1 = u1_tensor->template flat<float>();
    auto outputv = v_tensor->template flat<float>();

    // Call the cuda kernel0 launcher
    CpForwardUnfusedKernelLauncher(input.data(),
        kernel0.data(), kernel1.data(), kernel2.data(),
        outputu0.data(), outputu1.data(), outputv.data()
        );
  }
};

REGISTER_KERNEL_BUILDER(Name("CpForwardUnfused").Device(DEVICE_GPU), CpForwardUnfusedOp);


    /* OP_REQUIRES(context, TensorShapeUtils::IsMatrix(tenK0.shape()), */
    /*     errors::InvalidArgument("CpForwardUnfusedOp expects Input B to be a Matrix (rank 2 Tensor)")); */
    /* OP_REQUIRES(context, tenU.shape().dim_size(3) == tenK0.shape().dim_size(0), */
    /*     errors::InvalidArgument("CpForwardUnfusedOp expects Input A, Mode 2 to be equal length with Input B Mode 0")); */
