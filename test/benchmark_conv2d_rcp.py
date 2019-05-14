import tensorflow as tf
import numpy as np
import os
import layers
import utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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
rank=k0_rank

U = tf.random.uniform(inshape)
K0 = tf.random.uniform(k0shape)
K1 = tf.random.uniform(k1shape)
KC = tf.random.uniform(k2shape)

K = tf.random.uniform((3,3,16,16))

normal_kernel = {"kernel" : K}

if __name__ == "__main__":

    CPbench = tf.test.Benchmark()
    rcp_op_module = tf.load_op_library('../Kernels/rcp_fused_nchw.so')

    with tf.Session() as sess:
        with tf.device('/device:GPU:0'):
            # Normal, undecomposed Conv2D
            V_normal = layers.conv2d(tf.reshape(U, orig_shape), normal_kernel,
                    data_format=data_format)
            CPbench.run_op_benchmark(sess, V_normal, name='TF_normal_op', min_iters=min_iters)

            # Use einsum to recompose kernel first, then conv2d
            V_einsum = tf.einsum('axr,byr,hwr->hwabxy', K0, K1, KC)
            V_einsum = tf.reshape(V_einsum, K.shape)
            V_einsum = tf.nn.conv2d(tf.reshape(U, orig_shape), V_einsum,
                    strides=stride, padding=padding, data_format=data_format)
            CPbench.run_op_benchmark(sess, V_einsum, name='TF_einsum_recomp_op', min_iters=min_iters)

            # Fully fused operation
            V_fused = rcp_op_module.conv2d_rcp_fused_nchw(U, K0, K1, KC)
            CPbench.run_op_benchmark(sess, V_fused, name='fused_op', min_iters=min_iters)

            # Original paper from OP (or closest I could get working
            # (N,S0,S1,H,W) * (S0,T0,R)
            V_orig = tf.tensordot(U, K0, axes=[[1],[0]])
            # (N,S1,H,W,T0,R)
            V_orig = tf.transpose(V_orig, perm = [5,0,1,2,3,4])
            # (R,N,S1,H,W,T0) * (S1,T1,R)
            V_orig = tf.einsum('rnshwx,syr->nrxyhw', V_orig, K1)
            # (N,R,T0,T1,H,W)
            V_orig = tf.reshape(V_orig, (N,rank,T0*T1,H,W))
            # (N,R,T,H,W)
            V_orig = tf.nn.conv3d(V_orig, tf.reshape(KC, [1, *k2shape, 1]),
                    strides=[1,1,1,1,1],
                    padding="SAME",
                    data_format="NCDHW")
            V_orig = tf.reshape(V_orig, orig_shape)
            CPbench.run_op_benchmark(sess, V_orig, name='TF_orig_op', min_iters=min_iters)

            # Sequenced Op with Einsum Partial Outer Products NCHW
            U_seq = tf.random.uniform(inshape)
            K0_seq = tf.random.uniform(k0shape)
            K1_seq = tf.random.uniform(k1shape)
            KC_seq = tf.random.uniform(k2shape)

            K3_seq=tf.einsum('axr,byr->rabxy', K0_seq , K1_seq )
            U0_seq=tf.einsum('nabhw,rabxy->nrxyhw', U_seq ,K3_seq )
            V_seq = tf.reshape(U0_seq, (N,rank,T0*T1,H,W))
            V_seq = tf.nn.conv3d(V_seq, tf.reshape(KC_seq, [1, *k2shape, 1]),
                    strides=[1,1,1,1,1],
                    padding="SAME",
                    data_format="NCDHW")
            V_seq = tf.reshape(V_seq, orig_shape)
            CPbench.run_op_benchmark(sess, V_seq, name='TF_seq_nchw_op', min_iters=min_iters)

            # Sequenced Op with Einsum Partial Outer Products NHWC
            # (1,32,32,4,4)
            U_seq = tf.random.uniform((1,32,32,4,4))
            K0_seq = tf.random.uniform(k0shape)
            K1_seq = tf.random.uniform(k1shape)
            KC_seq = tf.random.uniform(k2shape)

            K3_seq=tf.einsum('axr,byr->rabxy', K0_seq , K1_seq )
            U0_seq=tf.einsum('nhwab,rabxy->nxyhwr', U_seq ,K3_seq )
            V_seq = tf.reshape(U0_seq, (N,T0*T1,H,W,rank))
            conv3dshape = [k2shape[0], k2shape[1], 1, rank, 1]
            V_seq = tf.nn.conv3d(V_seq, tf.reshape(KC_seq, conv3dshape),
                    strides=[1,1,1,1,1],
                    padding="SAME",
                    data_format="NDHWC")
            V_seq = tf.reshape(V_seq, U_seq.shape)
            CPbench.run_op_benchmark(sess, V_seq, name='TF_seq_nhwc_op', min_iters=min_iters)
