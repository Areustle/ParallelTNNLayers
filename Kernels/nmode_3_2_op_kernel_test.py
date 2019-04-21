import tensorflow as tf

test_module = tf.load_op_library('./nmode_3_2_op_kernel.so')

with tf.Session(''):
  result = test_module.n_mode32(
      [[[1, 2], [3, 4]],[[1,2],[3,4]]],
      [[1,2],[3,4]]).eval()
  print(result)
  result = test_module.n_mode32(
      [[[1, 2], [3, 4]],[[1,2],[3,4]]],
      [[1,2],[3,4]]).eval()
  print(result)
