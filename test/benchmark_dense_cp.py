import tensorflow as tf
import numpy as np
import os
import layers
import utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
# CP dense decomp matrix sizes:
# {'input_shape': [16, 16, 16], 'output_shape': [4, 4, 4], 'rank': 137}
# M:  (4096, 64)
# 0 (137, 16, 4)
# 1 (137, 16, 4)
# 2 (137, 16, 4)

N, S0, S1, S2 = inshape = (1, 16, 16, 16)
K0s, T0, rank0 = kshape0 = (16, 4, 137)
K1s, T1, rank1 = kshape1 = (16, 4, 137)
K2s, T2, rank2 = kshape2 = (16, 4, 137)
rank = rank0


# u_np  = np.random.uniform(size=inshape).astype(np.float32)
# k0_np = np.random.uniform(size=kshape0).astype(np.float32)
# k1_np = np.random.uniform(size=kshape1).astype(np.float32)
# k2_np = np.random.uniform(size=kshape2).astype(np.float32)

U = tf.random.uniform(inshape)
K0 = tf.random.uniform(kshape0)
K1 = tf.random.uniform(kshape1)
K2 = tf.random.uniform(kshape2)

M = tf.random.uniform((S0*S1*S2 , T0*T1*T2))

# m_np = np.einsum('axr,byr,czr->abcxyz', k0_np, k1_np, k2_np)
# m_np = np.reshape(m_np, (S0*S1*S2 , T0*T1*T2))

# v_np = np.einsum('ij,ni->nj', m_np, u_np.reshape(N, S0*S1*S2))
# v_np = np.reshape(v_np, (N, T0,T1,T2))

# w_np = np.einsum('nijk,ixr,jyr,kzr->nxyz', u_np, k0_np, k1_np, k2_np)

# print(np.allclose(w_np, v_np))
# v_np = np.einsum('nijk,ixr,jyr,kzr->nxyz', u_np, k0_np, k1_np, k2_np)


if __name__ == "__main__":

    CPbench = tf.test.Benchmark()
    cp_op_module = tf.load_op_library('../Kernels/dense_cp.so')

    with tf.Session() as sess:
        with tf.device('/device:GPU:0'):
            V_normal = tf.reshape(U, (N, S0*S1*S2))
            # M_normal = tf.einsum('axr,byr,czr->abcxyz', K0, K1, K2)
            # M_normal = tf.reshape(M_normal, (S0*S1*S2 , T0*T1*T2))
            V_normal = tf.einsum('ij,ni->nj', M, V_normal)
            V_normal = tf.reshape(V_normal, (N, T0,T1,T2))

            V_einsum = tf.einsum('nijk,ixr,jyr,kzr->nxyz', U, K0, K1, K2)

            # V_orig = layers.d_cp(U, cp_kernels, data_format=data_format)
            # V_custom = cp_op_module.conv2d_cp_fused_nchw(U, K0, K1, K2)
            V_custom = cp_op_module.dense_cp(U, K0, K1, K2)

            CPbench.run_op_benchmark(sess, V_normal, name='TF_normal_op', min_iters=1024)
            CPbench.run_op_benchmark(sess, V_custom, name='custom_op', min_iters=1024)
            CPbench.run_op_benchmark(sess, V_einsum, name='TF_einsum_op', min_iters=1024)

