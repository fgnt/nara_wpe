"""
Run individual test file with i.e.
    python nara_wpe/tests/test_tf_wpe.py
"""
import numpy as np
from nara_wpe import wpe
from nara_wpe import tf_wpe
import tensorflow as tf
from nara_wpe.test_utils import retry


class TestWPE(tf.test.TestCase):
    def setUp(self):
        self.T = np.random.randint(100, 120)
        self.D = np.random.randint(2, 8)
        self.K = np.random.randint(3, 5)
        self.delay = np.random.randint(0, 2)
        self.Y = np.random.normal(size=(self.D, self.T)) \
            + 1j * np.random.normal(size=(self.D, self.T))

    def test_inverse_power(self):
        np_inv_power = wpe.get_power_inverse(self.Y)

        with self.test_session() as sess:
            tf_signal = tf.placeholder(tf.complex128, shape=[None, None])
            tf_res = tf_wpe.get_power_inverse(tf_signal)
            tf_inv_power = sess.run(tf_res, {tf_signal: self.Y})

        np.testing.assert_allclose(np_inv_power, tf_inv_power)

    def test_correlations(self):
        np_inv_power = wpe.get_power_inverse(self.Y)
        np_corr = wpe.get_correlations_narrow_v5(
            self.Y, np_inv_power, self.K, self.delay
        )

        with tf.Graph().as_default(), tf.Session() as sess:
            tf_signal = tf.placeholder(tf.complex128, shape=[None, None])
            tf_inverse_power = tf_wpe.get_power_inverse(tf_signal)
            tf_res = tf_wpe.get_correlations_narrow(
                tf_signal, tf_inverse_power, self.K, self.delay
            )
            tf_corr = sess.run(tf_res, {tf_signal: self.Y})

        np.testing.assert_allclose(np_corr[0], tf_corr[0])
        np.testing.assert_allclose(np_corr[1], tf_corr[1])

    @retry(5)
    def test_filter_matrix(self):
        np_inv_power = wpe.get_power_inverse(self.Y)
        np_filter_matrix = wpe.get_filter_matrix_conj_v5(
            self.Y, np_inv_power, self.K, self.delay
        )

        with tf.Graph().as_default(), tf.Session() as sess:
            tf_signal = tf.placeholder(tf.complex128, shape=[None, None])
            tf_inverse_power = tf_wpe.get_power_inverse(tf_signal)
            tf_matrix, tf_vector = tf_wpe.get_correlations_narrow(
                tf_signal, tf_inverse_power, self.K, self.delay
            )
            tf_filter = tf_wpe.get_filter_matrix_conj(
                tf_signal, tf_inverse_power, tf_matrix, tf_vector,
                self.K, self.delay
            )
            tf_filter_matrix, tf_inv_power_2 = sess.run(
                [tf_filter, tf_inverse_power], {tf_signal: self.Y}
            )

        np.testing.assert_allclose(np_inv_power, tf_inv_power_2)
        np.testing.assert_allclose(np_filter_matrix, tf_filter_matrix)

    @retry(5)
    def test_filter_operation(self):
        np_inv_power = wpe.get_power_inverse(self.Y)
        np_filter_matrix = wpe.get_filter_matrix_conj_v5(
            self.Y, np_inv_power, self.K, self.delay
        )
        np_filter_op = wpe.perform_filter_operation_v4(
            self.Y, np_filter_matrix, self.K, self.delay
        )

        with tf.Graph().as_default(), tf.Session() as sess:
            tf_signal = tf.placeholder(tf.complex128, shape=[None, None])
            tf_inverse_power = tf_wpe.get_power_inverse(tf_signal)
            tf_matrix, tf_vector = tf_wpe.get_correlations_narrow(
                tf_signal, tf_inverse_power, self.K, self.delay
            )
            tf_filter = tf_wpe.get_filter_matrix_conj(
                tf_signal, tf_inverse_power, tf_matrix, tf_vector,
                self.K, self.delay
            )
            tf_filter_op = tf_wpe.perform_filter_operation(
                tf_signal, tf_filter, self.K, self.delay
            )
            tf_filter_op = sess.run(tf_filter_op, {tf_signal: self.Y})

        np.testing.assert_allclose(np_filter_op, tf_filter_op)


if __name__ == '__main__':
    tf.test.main()
