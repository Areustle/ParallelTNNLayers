import tensorflow as tf
import numpy as np
import os
import layers
import utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
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

U = tf.random.uniform(inshape)
K0 = tf.random.uniform(kshape0)
K1 = tf.random.uniform(kshape1)
K2 = tf.random.uniform(kshape2)

M = tf.random.uniform((S0*S1*S2 , T0*T1*T2))

if __name__ == "__main__":

    CPbench = tf.test.Benchmark()
    cp_op_module = tf.load_op_library('../Kernels/dense_cp.so')

    with tf.Session() as sess:
        with tf.device('/device:GPU:0'):
            V_gemm = tf.reshape(U, (N, S0*S1*S2))
            V_gemm = tf.matmul(V_gemm, M)
            # V_gemm = tf.einsum('ij,ni->nj', M, V_gemm)
            V_gemm = tf.reshape(V_gemm, (N, T0,T1,T2))

            M_rebuild = tf.einsum('axr,byr,czr->abcxyz', K0, K1, K2)
            M_rebuild = tf.reshape(M_rebuild, (S0*S1*S2 , T0*T1*T2))
            V_rebuild = tf.reshape(U, (N, S0*S1*S2))
            V_rebuild = tf.einsum('ij,ni->nj', M_rebuild, V_rebuild)
            V_rebuild = tf.reshape(V_rebuild, (N,T0,T1,T2))

            V_einsum = tf.einsum('nijk,ixr,jyr,kzr->nxyz', U, K0, K1, K2)

            V_fused = cp_op_module.dense_cp(U, K0, K1, K2)

            CPbench.run_op_benchmark(sess, V_gemm, name='TF_GEMM_op', min_iters=1024)
            CPbench.run_op_benchmark(sess, V_rebuild, name='TF_rebuild_op', min_iters=1024)
            CPbench.run_op_benchmark(sess, V_einsum, name='TF_einsum_op', min_iters=1024)
            CPbench.run_op_benchmark(sess, V_fused, name='custom_fused_op', min_iters=1024)

