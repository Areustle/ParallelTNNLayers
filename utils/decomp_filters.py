import tensorflow as tf
import numpy as np
from tensorly.decomposition import parafac, tucker, partial_tucker
import os
import utils
import layers

def decomp_svd_conv2d_nhwc(U, K, rate, verbose=True):
    kernel_size, kernel_size, input_filters, output_filters = K.shape

    params = layers.generate_params_conv2d_svd(input_filters, output_filters, kernel_size, rate)
    factors = utils.factorize_conv2d_svd(K, params)

    # if verbose:
    #     print("\nconv2d_svd decomp kernel sizes:")
    #     print(params)
    #     print("K:  {}".format(K.shape))
    #     for i, factor in enumerate(factors):
    #         print(" K{}: {}".format(i, factor.shape))

    kernels = {}
    kernels["kernel_0"] = factors[0]
    kernels["kernel_1"] = factors[1]

    layers.conv2d_svd(U, kernels, data_format="NCHW")
    return kernels









def decomp_cp_conv2d_nhwc(K, rate, verbose=True):
    kernel_size, kernel_size, input_filters, output_filters = K.shape

    params = layers.generate_params_conv2d_cp(input_filters, output_filters, kernel_size, rate)
    factors = utils.factorize_conv2d_cp(K, params)

    # if verbose:
    #     print("\nconv2d_cp decomp kernel sizes:")
    #     print(params)
    #     print("K:  {}".format(K.shape))
    #     for i, factor in enumerate(factors):
    #         print(" K{}: {}".format(i, factor.shape))

    kernels = {}
    kernels["kernel_0"] = factors[0]
    kernels["kernel_1"] = factors[1]
    kernels["kernel_2"] = factors[2]

    return kernels

def decomp_tk_conv2d_nhwc(K, rate, verbose=True):
    kernel_size, kernel_size, input_filters, output_filters = K.shape

    params = layers.generate_params_conv2d_tk(input_filters, output_filters, kernel_size, rate)
    factors = utils.factorize_conv2d_tk(K, params)

    if verbose:
        print("\nconv2d_tk decomp kernel sizes:")
        print(params)
        print("K:  {}".format(K.shape))
        for i, factor in enumerate(factors):
            print(" K{}: {}".format(i, factor.shape))

    kernels = {}
    kernels["kernel_0"] = factors[0]
    kernels["kernel_1"] = factors[1]
    kernels["kernel_2"] = factors[2]

    return kernels

def decomp_tt_conv2d_nhwc(K, rate, verbose=True):
    kernel_size, kernel_size, input_filters, output_filters = K.shape

    params = layers.generate_params_conv2d_tt(input_filters, output_filters, kernel_size, rate)
    factors = utils.factorize_conv2d_tt(K, params)

    if verbose:
        print("\nconv2d_tt decomp kernel sizes:")
        print(params)
        print("K:  {}".format(K.shape))
        for i, factor in enumerate(factors):
            print(" K{}: {}".format(i, factor.shape))

    kernels = {}
    kernels["kernel_0"] = factors[0]
    kernels["kernel_1"] = factors[1]
    kernels["kernel_2"] = factors[2]

    return kernels


def decomp_rcp_conv2d_nhwc(U, K, rate, verbose=True):
    kernel_size, kernel_size, input_filters, output_filters = K.shape

    params = layers.generate_params_conv2d_rcp(input_filters, output_filters, kernel_size, rate)
    dense_factors, conv_factor = utils.factorize_conv2d_rcp(K, params)


    kernels = {
            "kernel_0" : dense_factors[0],
            "kernel_1" : dense_factors[1],
          }

    kernels["kernel_conv"] = conv_factor

    print(type(kernels["kernel_0"]))

    layers.conv2d_rcp(U, kernels, data_format="NCHW")
    # if verbose:
    #     print("\nconv2d_rcp decomp kernel sizes:")
    #     print(params)
    #     print("K:  {}".format(K.shape))
    #     for i, factor in enumerate(dense_factors):
    #         print(" K{}: {}".format(i, factor.shape))
    #     print(" Conv_K: {}".format(conv_factor.shape))

    # return dense_factors, conv_factor

def decomp_rtk_conv2d_nhwc(K, rate, verbose=True):
    kernel_size, kernel_size, input_filters, output_filters = K.shape

    params = layers.generate_params_conv2d_rtk(input_filters, output_filters, kernel_size, rate)
    input_factors, core_factor, output_factors = utils.factorize_conv2d_rtk(K, params)

    if verbose:
        print("\nconv2d_rtk decomp kernel sizes:")
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
        print("\nconv2d_rtt decomp kernel sizes:")
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
        print("\nCP dense decomp matrix sizes:")
        print(params)
        print("M:  {}".format(M.shape))
        for i, factor in enumerate(factors):
            print(" M{}: {}".format(i, factor.shape))

    return factors


def decomp_tk_dense_nhwc(U, M, rate, verbose=True):
    xdim, ydim = M.shape

    params = layers.generate_params_dense_tk(xdim, ydim, rate)
    input_factors, core_factor, output_factors = utils.factorize_dense_tk(M, params)

    kernels = {}
    for i, factor in enumerate(input_factors):
        kernels["input_kernel_{}".format(i)] = factor

    kernels["core_kernel"] = core_factor
    for i, factor in enumerate(output_factors):
        kernels["output_kernel_{}".format(i)] = factor

    if verbose:
        print("\nTK dense decomp matrix sizes:")

    layers.dense_tk(U, kernels)

    #     print(params)
    #     print("M:  {}".format(M.shape))
    #     for i, factor in enumerate(input_factors):
    #         print(" input M{}: {}".format(i, factor.shape))
    #     print(" core M: {}".format(factor.shape))
    #     for i, factor in enumerate(output_factors):
    #         print(" output M{}: {}".format(i, factor.shape))

    # return input_factors, core_factor, output_factors


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

if __name__ == "__main__":

    U = np.random.normal(0., 1., [8,16,32,32]).astype(np.float32)
    K = np.random.normal(0., 1., [3,3,16,16]).astype(np.float32)
    # decomp_svd_conv2d_nhwc(U, K, 0.1)
    # decomp_cp_conv2d_nhwc(U, K, 0.1)
    # decomp_tk_conv2d_nhwc(U, K, 0.1)
    # decomp_tt_conv2d_nhwc(U, K, 0.1)

    decomp_rcp_conv2d_nhwc(U, K, 0.1)
    # decomp_rtk_conv2d_nhwc(U, K, 0.1)
    # decomp_rtt_conv2d_nhwc(U, K, 0.1)

    M = np.random.normal(0., 1., [128,128]).astype(np.float32)
    # decomp_cp_dense_nhwc(M, 0.1)
    # decomp_tk_dense_nhwc(U, M, 0.1)

