import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

class NModeTest(tf.test.TestCase):
  def testNMode(self):
    test_module = tf.load_op_library('./nmode_3_2_op_kernel.so')
    with self.session():
      result = test_module.n_mode32(
          [[[1, 2], [3, 4]],[[1,2],[3,4]]],
          [[1,2],[3,4]]).eval()
      expected = [[[ 7., 10.], [15., 22.]], [[ 7., 10.], [15., 22.]]]
      self.assertAllEqual(result, expected)

if __name__ == "__main__":
  tf.test.main()
