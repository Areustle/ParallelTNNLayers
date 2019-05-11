import tensorflow as tf
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

class CPOpTest(tf.test.TestCase):
    def testConv2dCpNchw(self):

        padding = "SAME"
        data_format = 'NCHW'
        U  = np.random.uniform(size=(8,16,32,32)).astype(np.float32)
        K0 = np.random.uniform(size=(16,6)).astype(np.float32)
        K1 = np.random.uniform(size=(3,3,6)).astype(np.float32)
        K2 = np.random.uniform(size=(6,16)).astype(np.float32)

        cp_op_module = tf.load_op_library('../Kernels/cp_fused_nchw.so')

        with self.session(force_gpu=True) as sess:

            V_custom = cp_op_module.conv2d_cp_fused_nchw(U, K0, K1, K2)

            V_orig = tf.nn.conv2d(U, K0.reshape(1,1,16,6), strides = [1, 1, 1, 1], padding = padding, data_format = data_format)
            V_orig = tf.nn.depthwise_conv2d(V_orig, K1.reshape(3,3,6,1), strides = [1, 1, 1, 1], padding = padding, data_format = data_format)
            V_orig = tf.nn.conv2d(V_orig, K2.reshape(1,1,6,16), strides = [1, 1, 1, 1], padding = padding, data_format = data_format)

            self.assertAllClose(V_custom.eval(), V_orig.eval(), rtol=1e-03, atol=1e-03)


if __name__ == "__main__":
    tf.test.main()
