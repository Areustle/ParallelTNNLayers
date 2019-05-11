import tensorflow as tf
import numpy as np
import os
import layers
import utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

padding = "SAME"
data_format = 'NCHW'
rate = 0.1
U  = np.random.uniform(size=(1,16,32,32)).astype(np.float32)
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

class CPOpTest(tf.test.TestCase):
    def testConv2dNormalNchw(self):
        with self.session(force_gpu=True) as sess:
            V_normal = layers.conv2d(U, normal_kernel, data_format=data_format)
            V_orig = layers.conv2d_cp(U, cp_kernels, data_format=data_format)
            self.assertAllClose(V_normal.eval(), V_orig.eval(), rtol=1e-03, atol=1e-03)

    def testConv2dCpNchw(self):
        cp_op_module = tf.load_op_library('../Kernels/cp_fused_nchw.so')
        with self.session(force_gpu=True) as sess:
            V_normal = layers.conv2d(U, normal_kernel, data_format=data_format)
            V_custom = cp_op_module.conv2d_cp_fused_nchw(U, K0, K1, K2)
            self.assertAllClose(V_normal.eval(), V_custom.eval(), rtol=1e-03, atol=1e-03)


if __name__ == "__main__":
    tf.test.main()
