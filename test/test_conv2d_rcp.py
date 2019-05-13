import tensorflow as tf
import numpy as np
import os
import layers
import utils
import tensorly as tl

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

min_iters = 1024
stride = [1,1,1,1]
padding = "SAME"
data_format = 'NCHW'
rate = 0.1

# "rcp_fused_nchw" : ((N,4,4,32,32),(4,4,11),(4,4,11),(3,3,11)),
N, S0, S1, H, W = inshape = (1,4,4,32,32)
KS0, T0, k0_rank = k0shape = (4,4,11)
KS1, T1, k1_rank = k1shape = (4,4,11)
kh, kw, k2_rank = k2shape = (3,3,11)
orig_shape = (1,16,32,32)

U = np.random.uniform(size=inshape).astype(np.float32)
K0 = np.random.uniform(size=k0shape).astype(np.float32)
K1 = np.random.uniform(size=k1shape).astype(np.float32)
KC = np.random.uniform(size=k2shape).astype(np.float32)

Kshape = (3,3,16,16)

class CPOpTest(tf.test.TestCase):
    def testConv2dRcpNchw(self):
        cp_op_module = tf.load_op_library('../Kernels/rcp_fused_nchw.so')
        with self.session(force_gpu=True) as sess:

            tfU = tf.convert_to_tensor(U)
            tfU = tf.reshape(tfU, orig_shape)
            tfK0 = tf.convert_to_tensor(K0)
            tfK1 = tf.convert_to_tensor(K1)
            tfKC = tf.convert_to_tensor(KC)

            V_einsum = tf.einsum('axr,byr,hwr->hwabxy', tfK0, tfK1, tfKC)
            V_einsum = tf.reshape(V_einsum, Kshape)
            V_einsum = tf.nn.conv2d(tfU, V_einsum,
                    strides=stride, padding=padding, data_format=data_format)
            V_einsum = tf.reshape(V_einsum, inshape)

            V_custom = cp_op_module.conv2d_rcp_fused_nchw(U, K0, K1, KC)

            self.assertAllClose(V_custom.eval(), V_einsum.eval(), rtol=1e-02, atol=1e-02)






if __name__ == "__main__":
    tf.test.main()

