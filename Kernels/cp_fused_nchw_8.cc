#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;  // NOLINT(build/namespaces)

REGISTER_OP("Conv2dCpFusedNchw")
    .Input("input: float")
    .Input("kernel0: float")
    .Input("kernel1: float")
    .Input("kernel2: float")
    .Output("output: float")
    .Doc(R"doc(Compute a 2D convolution operation using a CP decomposed convolution kernel.)doc");

/* Parameters: */
/* =========== */

/*   input: A 4th order Data tensor in NCHW format: */
/*       [batch, in_channels, in_height, in_width]. */

/*   kernel0: A 2nd order Kernel tensor [in_channels, rank]. */

/*   kernel1: A 3rd order Kernel tensor [filter_height, filter_width, rank] */

/*   kernel2: A 2nd order Kernel tensor [rank, out_channels] */

/* Results: */
/* ======== */

/*   output: A 4th order tensor in NCWH format: */
/*       [batch, out_channels, out_height, out_width] */


void Conv2dCpFusedNchwKernelLauncher(const float* U, const float* K0, const float* K1, const float* K2, float* V);

class Conv2dCpFusedOp : public OpKernel {
 public:
  explicit Conv2dCpFusedOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensors
    const Tensor& tenU = context->input(0);
    const Tensor& tenK0 = context->input(1);
    const Tensor& tenK1 = context->input(2);
    const Tensor& tenK2 = context->input(3);
    OP_REQUIRES(context, tenU.shape().dims()==4,
        errors::InvalidArgument("Conv2dCpFusedOp expects Input image to be a rank 4 Tensor NCWH"));
    OP_REQUIRES(context, tenK0.shape().dims()==2,
        errors::InvalidArgument("Conv2dCpFusedOp expects Kernel 0 to be a rank 2 Tensor [KchIN, rank]"));
    OP_REQUIRES(context, tenK1.shape().dims()==3,
        errors::InvalidArgument("Conv2dCpFusedOp expects Kernel 1 to be a rank 3 Tensor [HWR]"));
    OP_REQUIRES(context, tenK2.shape().dims()==2,
        errors::InvalidArgument("Conv2dCpFusedOp expects Kernel 2 to be a rank 2 Tensor [rank, KchoOUT]"));

    OP_REQUIRES(context, tenU.shape().dim_size(0)==8,
        errors::InvalidArgument("input[0] != 8"));
    OP_REQUIRES(context, tenU.shape().dim_size(1)==16,
        errors::InvalidArgument("input[1] != 32"));
    OP_REQUIRES(context, tenU.shape().dim_size(2)==32,
        errors::InvalidArgument("input[2] != 32"));
    OP_REQUIRES(context, tenU.shape().dim_size(3)==32,
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
    /* OP_REQUIRES(context, tenK1.shape().dim_size(3)==1, */
    /*     errors::InvalidArgument("kernel1[3] != 1")); */

    OP_REQUIRES(context, tenK2.shape().dim_size(0)==6,
        errors::InvalidArgument("kernel2[0] != 6"));
    OP_REQUIRES(context, tenK2.shape().dim_size(1)==16,
        errors::InvalidArgument("kernel2[1] != 16"));


    //  Create u0 output tensor
    Tensor* v_tensor = nullptr;
    auto v_outshape = tenU.shape();
    OP_REQUIRES_OK(context, context->allocate_output(0, v_outshape, &v_tensor));

    auto input = tenU.flat<float>();
    auto kernel0 = tenK0.flat<float>();
    auto kernel1 = tenK1.flat<float>();
    auto kernel2 = tenK2.flat<float>();
    auto output = v_tensor->template flat<float>();

    // Call the cuda kernel0 launcher
    Conv2dCpFusedNchwKernelLauncher(input.data(),
        kernel0.data(), kernel1.data(), kernel2.data(),
        output.data()
        );
  }
};

REGISTER_KERNEL_BUILDER(Name("Conv2dCpFusedNchw").Device(DEVICE_GPU), Conv2dCpFusedOp);

