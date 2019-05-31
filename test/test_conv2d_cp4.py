import tensorflow as tf
import numpy as np
import os
import layers
import utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

padding = "SAME"
data_format = 'NCHW'

inshape = (1,16,32,32)
k0shape = (16,6)
k1shape = (3,6)
k2shape = (3,6)
k3shape = (16,6)
rank=6

U  = np.random.uniform(size=inshape).astype(np.float32)
K0 = np.random.uniform(size=k0shape).astype(np.float32)
K1 = np.random.uniform(size=k1shape).astype(np.float32)
K2 = np.random.uniform(size=k2shape).astype(np.float32)
K3 = np.random.uniform(size=k3shape).astype(np.float32)

Kcp = np.einsum('ir,hr,wr,or->hwio', K0, K1, K2, K3)


class CPOpTest(tf.test.TestCase):
    def testConv2dNormalNchw(self):
        cp_op_module = tf.load_op_library('../Kernels/cp4_conv_nchw.so')
        with self.session(force_gpu=True) as sess:
            V_normal = tf.nn.conv2d(U, Kcp, strides=[1,1,1,1], padding='SAME', data_format=data_format)
            V_custom = cp_op_module.cp4_conv2d_nchw(U, K0, K1, K2, K3)
            self.assertAllClose(V_normal.eval(), V_custom.eval(), rtol=1e-03, atol=1e-03)


if __name__ == "__main__":
    tf.test.main()
