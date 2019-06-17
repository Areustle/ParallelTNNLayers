import tensorflow as tf
import numpy as np
import os
import layers
import utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

min_iters = 50000
padding = "SAME"
data_format = 'NCHW'

inshape = (1,16,32,32)
k0shape = (16,6)
k1shape = (3,6)
k2shape = (3,6)
k3shape = (16,6)
rank=6

U = np.random.uniform(size=inshape).astype(np.float)
K0 = np.random.uniform(size=k0shape).astype(np.float)
K1 = np.random.uniform(size=k1shape).astype(np.float)
K2 = np.random.uniform(size=k2shape).astype(np.float)
K3 = np.random.uniform(size=k3shape).astype(np.float)

K = np.einsum('ir,hr,wr,or->hwio', K0, K1, K2, K3)

if __name__ == "__main__":

    cp_op_module = tf.load_op_library('../Kernels/cp4_conv_nchw.so')

    with tf.Session() as sess:

        profiler = tf.profiler.Profiler(sess.graph)

        with tf.device('/device:GPU:0'):

            # Custom fused GPU implementation.
            V_fused = cp_op_module.cp4_conv2d_nchw(U, K0, K1, K2, K3)

            V_rebuild = tf.nn.conv2d(U, K, strides=[1,1,1,1], padding="SAME", data_format=data_format)

            run_meta = tf.RunMetadata()
            _ = sess.run(V_fused,
                    options = tf.RunOptions(
                        trace_level=tf.RunOptions.FULL_TRACE),
                    run_metadata=run_meta)
            profiler.add_step(0, run_meta)
            # # Profile the parameters of your model.
            profiler.profile_name_scope(options=(tf.profiler.ProfileOptionBuilder
                .trainable_variables_parameter()))

            # # Or profile the timing of your model operations.
            opts = tf.profiler.ProfileOptionBuilder.time_and_memory()
            profiler.profile_operations(options=opts)

        profiler.advise()

