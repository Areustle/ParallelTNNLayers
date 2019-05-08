import tensorflow as tf
import numpy as np
from tensorly.decomposition import parafac
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

def create_op_pairs():
    # Kernel Dimensions
    input_filters = 16
    output_filters = 16
    kernel_size = 3
    kernel_shape = [kernel_size, kernel_size, input_filters, output_filters]

    rate = 0.1

    # generate_params_conv2d_cp(input_filters, output_filters, kernel_size, rate):
    original_size = kernel_size * kernel_size * input_filters * output_filters
    unit_size = kernel_size * kernel_size + input_filters + output_filters

    rank = (rate * original_size) / unit_size
    rank = np.int(np.ceil(rank))
    #################

    # Generate a random convolution kernel
    K = np.random.normal(0., 1., kernel_shape).astype(np.float32)

    shape = K.shape
    assert len(shape) == 4, "The input tensor should be 4-order."

    # CP decompose the random convolution kernel
    K = np.moveaxis(K, 2, 0)
    K = np.reshape(K, (shape[2], shape[0] * shape[1], shape[3]))
    factors = parafac(K, rank)

    K0 = factors[0]
    K1 = np.reshape(factors[1], (shape[0], shape[1], rank, 1))
    K2 = np.reshape(np.transpose(factors[2]), (1, 1, rank, shape[3]))
    K2 = np.reshape(K2, (6, 16))
    U = np.random.normal(0., 1., [8,32,32,16]).astype(np.float32)

    return U, K0, K1, K2

if __name__ == "__main__":

    CPbench = tf.test.Benchmark()
    U, K0, K1, K2 = create_op_pairs()
    # cp_op_module = tf.load_op_library('../Kernels/cp_forward_unfused.so')
    cp_op_module = tf.load_op_library('../Kernels/cp_0_nhwc.so')

    with tf.Session() as sess:
        with tf.device('/device:GPU:0'):
            padding = "SAME"
            data_format = 'NHWC'

            V_orig = tf.nn.conv2d(U, K0.reshape(1,1,16,6), strides = [1, 1, 1, 1], padding = padding, use_cudnn_on_gpu=True, data_format = data_format)
            # V_orig = tf.nn.depthwise_conv2d(V_orig, K1, strides = [1, 1, 1, 1], padding = padding, data_format = data_format)
            # V_orig = tf.nn.conv2d(V_orig, K2.reshape(1,1,6,16), strides = [1, 1, 1, 1], padding = padding, use_cudnn_on_gpu=False, data_format = data_format)

            CPbench.run_op_benchmark(sess, V_orig, name='TF_op', min_iters=100)

            V_custom = cp_op_module.cp_forward_unfused(U, K0)

            CPbench.run_op_benchmark(sess, V_custom, name='custom_op', min_iters=100)

