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
U = np.random.uniform(size=(1,16,32,32)).astype(np.float32)
K = np.random.uniform(size=(3,3,16,16)).astype(np.float32)

kernel_size, kernel_size, input_filters, output_filters = K.shape

params = layers.generate_params_conv2d_cp(input_filters, output_filters, kernel_size, rate)
factors = utils.factorize_conv2d_cp(K, params)

cp_kernels = {}
cp_kernels["kernel_0"] = factors[0]
cp_kernels["kernel_1"] = factors[1]
cp_kernels["kernel_2"] = factors[2]
K0 = factors[0].reshape(16,6)
K1 = factors[1].reshape(3,3,6)
K2 = factors[2].reshape(6,16)

Kcp = utils.recompose_conv2d_cp(factors, params)
normal_kernel = {"kernel" : Kcp}


if __name__ == "__main__":

    CPbench = tf.test.Benchmark()
    cp_op_module = tf.load_op_library('../Kernels/cp_fused_nchw.so')

    with tf.Session() as sess:
        with tf.device('/device:GPU:0'):

            V_normal = layers.conv2d(U, normal_kernel, data_format=data_format)
            CPbench.run_op_benchmark(sess, V_normal, name='TF_normal_op', min_iters=min_iters)

            V_orig = layers.conv2d_cp(U, cp_kernels, data_format=data_format)
            CPbench.run_op_benchmark(sess, V_orig, name='TF_cp_op', min_iters=min_iters)


            V_fused = cp_op_module.conv2d_cp_fused_nchw(U,
                    K0.reshape(16,6),
                    K1.reshape(3,3,6),
                    K2.reshape(6,16))
            CPbench.run_op_benchmark(sess, V_fused, name='custom_fused_op', min_iters=min_iters)


            tU = tf.convert_to_tensor(U)
            tK0 = tf.convert_to_tensor(K0)
            tK1 = tf.convert_to_tensor(K1)
            tK2 = tf.convert_to_tensor(K2)

            V_seq_k3 = tf.einsum('hwr,rc->hwrc', tK1, tK2)
            V_seq_u0 = tf.einsum('nchw,cr->nrhw', tU, tK0)
            V_seq = tf.nn.conv2d(V_seq_u0, V_seq_k3, strides=[1,1,1,1], padding="SAME", data_format=data_format)
            CPbench.run_op_benchmark(sess, V_seq, name='sequencer_op', min_iters=min_iters)
            # V_seq_k3 = tf.einsum('hwr,rc->hwrc', tK1, tK2)
            # V_seq_u0 = tf.einsum('nchw,cr->nhwr', tU, tK0)
            # V_seq = tf.nn.conv2d(V_seq_u0, V_seq_k3, strides=[1,1,1,1], padding="SAME")

