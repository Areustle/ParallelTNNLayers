import tensorflow as tf
import numpy as np
import os
import layers
import utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

padding = "SAME"
data_format = 'NCHW'

U  = np.random.uniform(size=(1,16,32,32)).astype(np.float32)
K0 = np.random.uniform(size=(1,1,16,6)).astype(np.float32)
K1 = np.random.uniform(size=(3,3,6,1)).astype(np.float32)
K2 = np.random.uniform(size=(1,1,6,16)).astype(np.float32)

cp_kernels = {"kernel_0" : K0, "kernel_1" : K1, "kernel_2" : K2}

Kcp = np.einsum('llar,ijrl,llrx->ijax', K0, K1, K2)
normal_kernel = {"kernel" : Kcp}
K0 = K0.reshape(16,6)
K1 = K1.reshape(3,3,6)
K2 = K2.reshape(6,16)


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

    def testConv2dSequencedNchw(self):
        with self.session(force_gpu=True) as sess:
            V_normal = layers.conv2d(U, normal_kernel, data_format=data_format)
            tU = tf.convert_to_tensor(U)
            tK0 = tf.convert_to_tensor(K0)
            tK1 = tf.convert_to_tensor(K1)
            tK2 = tf.convert_to_tensor(K2)

            V_seq_k3 = tf.einsum('hwr,rc->hwrc', tK1, tK2)
            V_seq_u0 = tf.einsum('nchw,cr->nrhw', tU, tK0)
            V_seq = tf.nn.conv2d(V_seq_u0, V_seq_k3, strides=[1,1,1,1], padding="SAME", data_format=data_format)
            self.assertAllClose(V_normal.eval(), V_seq.eval(), rtol=1e-03, atol=1e-03)

    def testConv2dSequencedNhwc(self):
        with self.session(force_gpu=True) as sess:
            V_normal = layers.conv2d(U, normal_kernel, data_format=data_format)
            tU = tf.convert_to_tensor(U)
            tK0 = tf.convert_to_tensor(K0)
            tK1 = tf.convert_to_tensor(K1)
            tK2 = tf.convert_to_tensor(K2)

            V_seq_k3 = tf.einsum('hwr,rc->hwrc', tK1, tK2)
            V_seq_u0 = tf.einsum('nchw,cr->nhwr', tU, tK0)
            V_seq = tf.nn.conv2d(V_seq_u0, V_seq_k3, strides=[1,1,1,1], padding="SAME")
            V_seq = tf.transpose(V_seq, [0,3,1,2])
            self.assertAllClose(V_normal.eval(), V_seq.eval(), rtol=1e-03, atol=1e-03)



if __name__ == "__main__":
    tf.test.main()
