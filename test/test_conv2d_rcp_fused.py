import tensorflow as tf
import numpy as np
import os
import layers

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

padding = "SAME"
data_format = 'NCHW'
U  = np.random.uniform(size=(8,16,32,32)).astype(np.float32)
K0 = np.random.uniform(size=(1,1,16,6)).astype(np.float32)
K1 = np.random.uniform(size=(3,3,6,1)).astype(np.float32)
K2 = np.random.uniform(size=(1,1,6,16)).astype(np.float32)

Kcp = utils.recompose_kernel_rcp(K0, K1, K2)

normal_kernel = {"kernel" : Kcp}
cp_kernels = { "kernel_0" : K0, "kernel_1" : K1, "kernel_2" : K2 }


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
            V_custom = cp_op_module.conv2d_cp_fused_nchw(U,
                    K0.reshape(16,6),
                    K1.reshape(3,3,6),
                    K2.reshape(6,16))
            self.assertAllClose(V_normal.eval(), V_custom.eval(), rtol=1e-03, atol=1e-03)


if __name__ == "__main__":
    tf.test.main()







#class CPOpTest(tf.test.TestCase):
#    def testConv2dRcpNchw(self):
#        cp_op_module = tf.load_op_library('../Kernels/rcp_fused_nchw.so')

#        config = tf.ConfigProto(allow_soft_placement=True,
#                log_device_placement=True,
#                device_count={'GPU': 1})
#        with self.session(force_gpu=True) as sess:

#            padding = "SAME"
#            data_format = 'NCHW'
#            data_format_3d = "NCDHW"
#            U  = tf.random.uniform((8,4,4,32,32))
#            K0 = tf.random.uniform((4,4,11))
#            K1 = tf.random.uniform((4,4,11))
#            KC = tf.random.uniform((3,3,11))


#            # Build TF OP metadata
#            order, rank = 2, 11
#            input_shape, output_shape, axes = [4,4], [4,4], [[1],[0]]
#            strides_3d = [1, 1, 1, 1, 1]
#            shape = [3,3,11] ## Conv Kernel Shape

#            kernels = {
#                    'kernel_0': tf.transpose(K0, perm=[2,0,1]),
#                    'kernel_1': tf.transpose(K1, perm=[2,0,1]),
#                    'kernel_conv':KC }


#            V_orig = layers.conv2d_rcp(tf.reshape(U, (8,16,32,32)), kernels, data_format="NCHW")
#            #########################
#            # Original TF Operation #
#            #########################

#            # kernel = tf.transpose(kernels["kernel_0"], perm = [1, 2, 0])
#            # tensor = tf.tensordot(U, kernel, axes = axes)
#            # tensor = tf.transpose(tensor, perm = [order + 3] + list(range(order + 3)))

#            # contract = lambda var: (tf.tensordot(var[0], var[1], axes = axes), 0)
#            # for l in range(1, order):
#            #     tensor, _ = tf.map_fn(contract, (tensor, kernels["kernel_" + str(l)]))
#            # tensor = tf.reshape(tensor, [rank] + [-1] + tensor.shape.as_list()[2:4] + [np.prod(output_shape)])
#            # tensor = tf.transpose(tensor, perm = [1, 0, 4, 2, 3])
#            # kernel = tf.reshape(kernels["kernel_conv"], [1, shape[0], shape[1], rank, 1])
#            # tensor = tf.nn.conv3d(tensor, kernel, strides = strides_3d, padding = padding,
#            #         data_format = data_format_3d)
#            # V_orig = tf.reshape(tensor, [-1] + [4,4,32,32])#tensor.shape.as_list()[2:5])
#            V_orig.eval()


#            ####################
#            # Custom Operation #
#            ####################
#            # V_custom = cp_op_module.conv2d_rcp_fused_nchw(U, K0, K1, KC)

#            ###################
#            # Test Operations #
#            ###################
#            # self.assertAllClose(V_custom.eval(), V_orig.eval(), rtol=1e-03, atol=1e-03)


