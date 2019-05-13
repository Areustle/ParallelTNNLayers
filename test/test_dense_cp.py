import tensorflow as tf
import numpy as np
import os
import layers
import utils
import tensorly as tl

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
# CP dense decomp matrix sizes:
# {'input_shape': [16, 16, 16], 'output_shape': [4, 4, 4], 'rank': 137}
# M:  (4096, 64)
# 0 (137, 16, 4)
# 1 (137, 16, 4)
# 2 (137, 16, 4)

# "rcp_fused_nchw" : ((N,4,4,32,32),(4,4,11),(4,4,11),(3,3,11)),
N, S0, S1, S2 = inshape = (1, 16, 16, 16)
K0s, T0, rank0 = kshape0 = (16, 4, 137)
K1s, T1, rank1 = kshape1 = (16, 4, 137)
K2s, T2, rank2 = kshape2 = (16, 4, 137)
rank = rank0
orig_shape = (1,16,32,32)

U = np.random.uniform(size=inshape).astype(np.float32)
K0 = np.random.uniform(size=kshape0).astype(np.float32)
K1 = np.random.uniform(size=kshape1).astype(np.float32)
K2 = np.random.uniform(size=kshape2).astype(np.float32)

class CPOpTest(tf.test.TestCase):
    def testDenseCp(self):
        dense_cp_op_module = tf.load_op_library('../Kernels/dense_cp.so')
        with self.session(force_gpu=True) as sess:

            tfU = tf.convert_to_tensor(U)
            tfK0 = tf.convert_to_tensor(K0)
            tfK1 = tf.convert_to_tensor(K1)
            tfK2 = tf.convert_to_tensor(K2)

            V_einsum = tf.einsum('nijk,ixr,jyr,kzr->nxyz', tfU, tfK0, tfK1, tfK2)

            V_custom = dense_cp_op_module.dense_cp(U, K0, K1, K2)

            self.assertAllClose(V_custom.eval(), V_einsum.eval(), rtol=1e-02, atol=1e-02)




if __name__ == "__main__":
    tf.test.main()

