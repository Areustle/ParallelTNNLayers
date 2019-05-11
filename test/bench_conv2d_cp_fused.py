import tensorflow as tf
import numpy as np
import os
import layers
import utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

padding = "SAME"
data_format = 'NCHW'
U  = np.random.uniform(size=(8,16,32,32)).astype(np.float32)
K0 = np.random.uniform(size=(1,1,16,6)).astype(np.float32)
K1 = np.random.uniform(size=(3,3,6,1)).astype(np.float32)
K2 = np.random.uniform(size=(1,1,6,16)).astype(np.float32)

Kcp = utils.recompose_kernel_cp(K0, K1, K2)

normal_kernel = {"kernel" : Kcp}
cp_kernels = { "kernel_0" : K0, "kernel_1" : K1, "kernel_2" : K2 }


if __name__ == "__main__":

    CPbench = tf.test.Benchmark()
    cp_op_module = tf.load_op_library('../Kernels/cp_fused_nchw.so')

    with tf.Session() as sess:
        with tf.device('/device:GPU:0'):
            V_normal = layers.conv2d(U, normal_kernel, data_format=data_format)
            V_orig = layers.conv2d_cp(U, cp_kernels, data_format=data_format)
            # V_custom = cp_op_module.conv2d_cp_fused_nchw(U, K0, K1, K2)
            V_custom = cp_op_module.conv2d_cp_fused_nchw(U,
                    K0.reshape(16,6),
                    K1.reshape(3,3,6),
                    K2.reshape(6,16))

            CPbench.run_op_benchmark(sess, V_custom, name='custom_op', min_iters=400)
            CPbench.run_op_benchmark(sess, V_normal, name='TF_normal_op', min_iters=400)
            CPbench.run_op_benchmark(sess, V_orig, name='TF_cp_op', min_iters=400)

