"""
Run all tests with:
    nosetests -w tests/
"""
import numpy as np
from nara_wpe import wpe
from nara_wpe import tf_wpe
import tensorflow as tf


class TestWPE(tf.test.TestCase):
    def test_inverse_power(self):
        D, T = 6, 300
        signal = (
            np.random.normal(0, 0.2, (D, T))
            + 1j * np.random.normal(0, 0.2, (D, T))
        )
        np_inv_power = wpe.get_power_inverse(signal)
        with self.test_session() as sess:
            tf_signal = tf.placeholder(tf.complex128, shape=[None, None])
            tf_res = tf_wpe.get_power_inverse(tf_signal)
            tf_inv_power = sess.run(tf_res, {tf_signal: signal})
        np.testing.assert_allclose(np_inv_power, tf_inv_power)


if __name__ == '__main__':
    tf.test.main()
