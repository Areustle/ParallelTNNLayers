import tensorflow as tf
import numpy as np
from tensorly.decomposition import parafac, tucker, partial_tucker
import os
import utils
import layers

def decomp_conv2d_nhwc(K, name, genpar, factor, rate):
    kernel_size, kernel_size, input_filters, output_filters = K.shape

    params = genpar(input_filters, output_filters, kernel_size, rate)
    factors = factor(K, params)

    print("{} decomp kernel sizes:".format(name))
    print("K:  {}".format(K.shape))
    for i, factor in enumerate(factors):
        print(" K{}: {}".format(i, factor.shape))
    return factors

def decomp_r_conv2d_nhwc(K, name, genpar, factor, rate):
    kernel_size, kernel_size, input_filters, output_filters = K.shape

    params = genpar(input_filters, output_filters, kernel_size, rate)
    factor_list = factor(K, params)

    print("{} decomp kernel sizes:".format(name))
    print("K:  {}".format(K.shape))
    for i, factors in enumerate(factor_list):
        for j, factor in enumerate(factors):
            print(" K{}{}: {}".format(i,j, factor.shape))

    return factor_list

def decomp_dense_nhwc(M, name, genpar, factor, rate):
    xdim, ydim = M.shape

    params = genpar(xdim, ydim, rate)
    factor_list = factor(M, params)

    print("{} decomp matrix sizes:".format(name))
    print("M:  {}".format(M.shape))
    for i, factors in enumerate(factor_list):
        for j, factor in enumerate(factors):
            print(" M{}{}: {}".format(i,j, factor.shape))


K = np.random.normal(0., 1., [3,3,16,16]).astype(np.float32)
decomp_conv2d_nhwc(K, 'conv2d_svd', layers.generate_params_conv2d_svd, utils.factorize_conv2d_svd, 0.1)
decomp_conv2d_nhwc(K, 'conv2d_cp', layers.generate_params_conv2d_cp, utils.factorize_conv2d_cp, 0.1)
decomp_conv2d_nhwc(K, 'conv2d_tk', layers.generate_params_conv2d_tk, utils.factorize_conv2d_tk, 0.1)
decomp_conv2d_nhwc(K, 'conv2d_tt', layers.generate_params_conv2d_tt, utils.factorize_conv2d_tt, 0.1)

decomp_r_conv2d_nhwc(K, 'conv2d_rcp', layers.generate_params_conv2d_rcp, utils.factorize_conv2d_rcp, 0.1)
decomp_r_conv2d_nhwc(K, 'conv2d_rtk', layers.generate_params_conv2d_rtk, utils.factorize_conv2d_rtk, 0.1)
decomp_r_conv2d_nhwc(K, 'conv2d_rtt', layers.generate_params_conv2d_rtt, utils.factorize_conv2d_rtt, 0.1)

M = np.random.normal(0., 1., [256,256]).astype(np.float32)
decomp_dense_nhwc(M, 'dense_cp', layers.generate_params_dense_cp, utils.factorize_dense_cp, 0.1)
# decomp_dense_nhwc(M, 'dense_tk', layers.generate_params_dense_tk, utils.factorize_dense_tk, 0.1)
# decomp_dense_nhwc(M, 'dense_tt', layers.generate_params_dense_tt, utils.factorize_dense_tt, 0.1)
