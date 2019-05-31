import tensorflow as tf
import numpy as np
import os
import layers
import utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

min_iters = 1024
padding = "SAME"
data_format = 'NCHW'

inshape = (1,16,32,32)
k0shape = (16,6)
k1shape = (3,6)
k2shape = (3,6)
k3shape = (16,6)
rank=6

U = tf.random.uniform(inshape)
K0 = tf.random.uniform(k0shape)
K1 = tf.random.uniform(k1shape)
K2 = tf.random.uniform(k2shape)
K3 = tf.random.uniform(k3shape)

if __name__ == "__main__":

    CPbench = tf.test.Benchmark()
    cp_op_module = tf.load_op_library('../Kernels/cp4_conv_nchw.so')

    with tf.Session() as sess:
        with tf.device('/device:GPU:0'):

            # Custom fused GPU implementation.
            V_fused = cp_op_module.cp4_conv2d_nchw(U, K0, K1, K2, K3)
            CPbench.run_op_benchmark(sess, V_fused, name='custom_fused_op', min_iters=min_iters)

            # Rebuild Op.
            V_rebuild = tf.einsum('ir,hr,wr,or->hwio', K0, K1, K2, K3)
            V_rebuild = tf.nn.conv2d(tU, V_rebuild, strides=[1,1,1,1], padding="SAME", data_format=data_format)
            CPbench.run_op_benchmark(sess, V_rebuild, name='TF_rebuild_nchw_op', min_iters=min_iters)
