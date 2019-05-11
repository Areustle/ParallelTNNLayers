import tensorflow as tf
import numpy as np
import os
import layers
import utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

min_iters = 1024
padding = "SAME"
data_format = 'NCHW'
rate = 0.1

U_rcp = np.random.uniform(size=(1,4,4,32,32)).astype(np.float32)
U_norm = np.reshape(U_rcp, (1,16,32,32))
K = np.random.uniform(size=(3,3,16,16)).astype(np.float32)

kernel_size, kernel_size, input_filters, output_filters = K.shape
params = layers.generate_params_conv2d_rcp(input_filters, output_filters, kernel_size, rate)
# factors = utils.factorize_conv2d_rcp(K, params)
dense_factors, conv_factor = utils.factorize_conv2d_rcp(K, params)
# print(params)

kernels = { "kernel_0" : dense_factors[0], "kernel_1" : dense_factors[1], }
kernels["kernel_conv"] = conv_factor

# for k,v in kernels.items():
#     print(k, v.shape)

K0 = dense_factors[0].reshape(4,4,11)
K1 = dense_factors[1].reshape(4,4,11)
KC = conv_factor.reshape(3,3,11)

K_recomp = utils.recompose_conv2d_rcp(dense_factors, conv_factor, params)
# print(K_recomp.shape)
normal_kernel = {"kernel" : K_recomp}


if __name__ == "__main__":

    CPbench = tf.test.Benchmark()
    cp_op_module = tf.load_op_library('../Kernels/rcp_fused_nchw.so')

    with tf.Session() as sess:
        with tf.device('/device:GPU:0'):
            V_normal = layers.conv2d(U_norm, normal_kernel, data_format=data_format)
            # V_orig = layers.conv2d_cp(U, cp_kernels, data_format=data_format)
            # print(K0.shape)
            # print(K1.shape)
            # print(KC.shape)
            V_fused = cp_op_module.conv2d_rcp_fused_nchw(U_rcp, K0, K1, KC)

            CPbench.run_op_benchmark(sess, V_fused, name='fused_op', min_iters=min_iters)
            CPbench.run_op_benchmark(sess, V_normal, name='TF_normal_op', min_iters=min_iters)
            # CPbench.run_op_benchmark(sess, V_orig, name='TF_cp_op', min_iters=min_iters)

