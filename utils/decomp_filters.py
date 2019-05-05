import tensorflow as tf
import numpy as np
from tensorly.decomposition import parafac, tucker, partial_tucker
import os
import utils
import layers

def decomp_svd_conv2d_nhwc(K, rate, verbose=True):
    kernel_size, kernel_size, input_filters, output_filters = K.shape

    params = layers.generate_params_conv2d_svd(input_filters, output_filters, kernel_size, rate)
    factors = utils.factorize_conv2d_svd(K, params)

    if verbose:
        print("conv2d_svd decomp kernel sizes:")
        print(params)
        print("K:  {}".format(K.shape))
        for i, factor in enumerate(factors):
            print(" K{}: {}".format(i, factor.shape))

    return factors

def decomp_cp_conv2d_nhwc(K, rate, verbose=True):
    kernel_size, kernel_size, input_filters, output_filters = K.shape

    params = layers.generate_params_conv2d_cp(input_filters, output_filters, kernel_size, rate)
    factors = utils.factorize_conv2d_cp(K, params)

    if verbose:
        print("conv2d_cp decomp kernel sizes:")
        print(params)
        print("K:  {}".format(K.shape))
        for i, factor in enumerate(factors):
            print(" K{}: {}".format(i, factor.shape))

    return factors

def decomp_tk_conv2d_nhwc(K, rate, verbose=True):
    kernel_size, kernel_size, input_filters, output_filters = K.shape

    params = layers.generate_params_conv2d_tk(input_filters, output_filters, kernel_size, rate)
    factors = utils.factorize_conv2d_tk(K, params)

    if verbose:
        print("conv2d_tk decomp kernel sizes:")
        print(params)
        print("K:  {}".format(K.shape))
        for i, factor in enumerate(factors):
            print(" K{}: {}".format(i, factor.shape))

    return factors

def decomp_tt_conv2d_nhwc(K, rate, verbose=True):
    kernel_size, kernel_size, input_filters, output_filters = K.shape

    params = layers.generate_params_conv2d_tt(input_filters, output_filters, kernel_size, rate)
    factors = utils.factorize_conv2d_tt(K, params)

    if verbose:
        print("conv2d_tt decomp kernel sizes:")
        print(params)
        print("K:  {}".format(K.shape))
        for i, factor in enumerate(factors):
            print(" K{}: {}".format(i, factor.shape))

    return factors

def decomp_rcp_conv2d_nhwc(K, rate, verbose=True):
    kernel_size, kernel_size, input_filters, output_filters = K.shape

    params = layers.generate_params_conv2d_rcp(input_filters, output_filters, kernel_size, rate)
    dense_factors, conv_factor = utils.factorize_conv2d_rcp(K, params)

    if verbose:
        print("conv2d_rcp decomp kernel sizes:")
        print(params)
        print("K:  {}".format(K.shape))
        for i, factor in enumerate(dense_factors):
            print(" K{}: {}".format(i, factor.shape))
        print(" Conv_K: {}".format(conv_factor.shape))

    return dense_factors, conv_factor

def decomp_rtk_conv2d_nhwc(K, rate, verbose=True):
    kernel_size, kernel_size, input_filters, output_filters = K.shape

    params = layers.generate_params_conv2d_rtk(input_filters, output_filters, kernel_size, rate)
    input_factors, core_factor, output_factors = utils.factorize_conv2d_rtk(K, params)

    if verbose:
        print("conv2d_rtk decomp kernel sizes:")
        print(params)
        print("K:  {}".format(K.shape))
        for i, factor in enumerate(input_factors):
            print(" input K{}: {}".format(i, factor.shape))
        print(" core K: {}".format(factor.shape))
        for i, factor in enumerate(output_factors):
            print(" output K{}: {}".format(i, factor.shape))

    return input_factors, core_factor, output_factors

def decomp_rtt_conv2d_nhwc(K, rate, verbose=True):
    kernel_size, kernel_size, input_filters, output_filters = K.shape

    params = layers.generate_params_conv2d_rtt(input_filters, output_filters, kernel_size, rate)
    factors, last_fact = utils.factorize_conv2d_rtt(K, params)

    if verbose:
        print("conv2d_rtt decomp kernel sizes:")
        print(params)
        print("K:  {}".format(K.shape))
        for i, factor in enumerate(factors):
            print(" K{}: {}".format(i, factor.shape))
        print(" Last K: {}".format(last_fact.shape))

    return factors

def decomp_cp_dense_nhwc(M, rate, verbose=True):
    xdim, ydim = M.shape

    params = layers.generate_params_dense_cp(xdim, ydim, rate)
    factors = utils.factorize_dense_cp(M, params)

    if verbose:
        print("CP dense decomp matrix sizes:")
        print(params)
        print("M:  {}".format(M.shape))
        for i, factor in enumerate(factors):
            print(" M{}: {}".format(i, factor.shape))

    return factors


def decomp_tk_dense_nhwc(M, rate, verbose=True):
    xdim, ydim = M.shape

    params = layers.generate_params_dense_tk(xdim, ydim, rate)
    input_factors, core_factor, output_factors = utils.factorize_dense_tk(M, params)

    if verbose:
        print("TK dense decomp matrix sizes:")
        print(params)
        print("M:  {}".format(M.shape))
        for i, factor in enumerate(input_factors):
            print(" input M{}: {}".format(i, factor.shape))
        print(" core M: {}".format(factor.shape))
        for i, factor in enumerate(output_factors):
            print(" output M{}: {}".format(i, factor.shape))

    return input_factors, core_factor, output_factors


# def decomp_tt_dense_nhwc(M, rate, verbose=True):
#     xdim, ydim = M.shape

#     params = layers.generate_params_dense_tt(xdim, ydim, rate)
#     print(params)
#     factors, last_fact = utils.factorize_dense_tt(M, params)

#     if verbose:
#         print("TT dense decomp matrix sizes:")
#         print(params)
#         # print("M:  {}".format(M.shape))
#         # for i, factor in enumerate(factors):
#         #     print(" M{}: {}".format(i, factor.shape))
#         # print(" Last M: {}".format(last_fact.shape))

#     # return factors, last_fact

K = np.random.normal(0., 1., [3,3,16,16]).astype(np.float32)
decomp_svd_conv2d_nhwc(K, 0.1)
decomp_cp_conv2d_nhwc(K, 0.1)
decomp_tk_conv2d_nhwc(K, 0.1)
decomp_tt_conv2d_nhwc(K, 0.1)

decomp_rcp_conv2d_nhwc(K, 0.1)
decomp_rtk_conv2d_nhwc(K, 0.1)
decomp_rtt_conv2d_nhwc(K, 0.1)

M = np.random.normal(0., 1., [256,256]).astype(np.float32)
decomp_cp_dense_nhwc(M, 0.1)
decomp_tk_dense_nhwc(M, 0.1)
# decomp_tt_dense_nhwc(M, 0.1)

