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
        np.random.seed(0)
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
        np_corr = wpe.get_correlations_narrow(
            self.Y, np_inv_power, self.K, self.delay
        )

        with tf.Graph().as_default(), tf.Session() as sess:
            tf_signal = tf.placeholder(tf.complex128, shape=[None, None])
            tf_inverse_power = tf_wpe.get_power_inverse(tf_signal)
            tf_res = tf_wpe.get_correlations_for_single_frequency(
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
            tf_matrix, tf_vector = tf_wpe.get_correlations_for_single_frequency(
                tf_signal, tf_inverse_power, self.K, self.delay
            )
            tf_filter = tf_wpe.get_filter_matrix_conj(
                tf_signal, tf_matrix, tf_vector,
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
            tf_matrix, tf_vector = tf_wpe.get_correlations_for_single_frequency(
                tf_signal, tf_inverse_power, self.K, self.delay
            )
            tf_filter = tf_wpe.get_filter_matrix_conj(
                tf_signal, tf_matrix, tf_vector,
                self.K, self.delay
            )
            tf_filter_op = tf_wpe.perform_filter_operation(
                tf_signal, tf_filter, self.K, self.delay
            )
            tf_filter_op = sess.run(tf_filter_op, {tf_signal: self.Y})

        np.testing.assert_allclose(np_filter_op, tf_filter_op)

    def test_wpe_step(self):
        with self.test_session() as sess:
            Y = tf.convert_to_tensor(self.Y[None])
            enhanced, inv_power = tf_wpe.single_frequency_wpe(
                Y[0], iterations=3
            )
            step_enhanced = tf_wpe.wpe_step(Y, inv_power[None])
            enhanced, step_enhanced = sess.run(
                [enhanced, step_enhanced]
            )
        np.testing.assert_allclose(enhanced, step_enhanced[0])

    def _get_batch_data(self):
        Y = tf.convert_to_tensor(self.Y[None])
        inv_power = tf_wpe.get_power_inverse(Y[0])[None]
        Y_short = Y[..., :self.T-20]
        inv_power_short = inv_power[..., :self.T-20]
        Y_batch = tf.stack(
            [Y, tf.pad(Y_short, ((0, 0), (0, 0), (0, 20)))]
        )
        inv_power_batch = tf.stack(
            [inv_power, tf.pad(inv_power_short, ((0, 0), (0, 20)))]
        )
        return Y_batch, inv_power_batch

    def test_batched_wpe_step(self):
        with self.test_session() as sess:
            Y_batch, inv_power_batch = self._get_batch_data()
            enhanced_ref_1 = tf_wpe.wpe_step(
                Y_batch[0], inv_power_batch[0]
            )
            enhanced_ref_2 = tf_wpe.wpe_step(
                Y_batch[0, ...,:self.T-20], inv_power_batch[0, ...,:self.T-20]
            )
            step_enhanced = tf_wpe.batched_wpe_step(
                Y_batch, inv_power_batch,
                num_frames=tf.convert_to_tensor([self.T, self.T-20])
            )
            enhanced, ref1, ref2 = sess.run(
                [step_enhanced, enhanced_ref_1, enhanced_ref_2]
            )
        np.testing.assert_allclose(enhanced[0], ref1)
        np.testing.assert_allclose(enhanced[1, ..., :-20], ref2)

    def test_wpe(self):
        with self.test_session() as sess:
            Y = tf.convert_to_tensor(self.Y)
            enhanced, inv_power = tf_wpe.single_frequency_wpe(
                Y, iterations=1
            )
            enhanced = sess.run(enhanced)
        ref = wpe.wpe_v7(self.Y, iterations=1, statistics_mode='valid')
        np.testing.assert_allclose(enhanced, ref)

    def test_batched_wpe(self):
        with self.test_session() as sess:
            Y_batch, _ = self._get_batch_data()
            enhanced_ref_1 = tf_wpe.wpe(Y_batch[0])
            enhanced_ref_2 = tf_wpe.wpe(Y_batch[0, ..., :self.T-20])
            step_enhanced = tf_wpe.batched_wpe(
                Y_batch,
                num_frames=tf.convert_to_tensor([self.T, self.T-20])
            )
            enhanced, ref1, ref2 = sess.run(
                [step_enhanced, enhanced_ref_1, enhanced_ref_2]
            )
        np.testing.assert_allclose(enhanced[0], ref1)
        np.testing.assert_allclose(enhanced[1, ..., :-20], ref2)

    def test_batched_block_wpe_step(self):
        with self.test_session() as sess:
            Y_batch, inv_power_batch = self._get_batch_data()
            enhanced_ref_1 = tf_wpe.block_wpe_step(
                Y_batch[0], inv_power_batch[0]
            )
            enhanced_ref_2 = tf_wpe.block_wpe_step(
                Y_batch[0, ..., :self.T-20], inv_power_batch[0, ..., :self.T-20]
            )
            step_enhanced = tf_wpe.batched_block_wpe_step(
                Y_batch, inv_power_batch,
                num_frames=tf.convert_to_tensor([self.T, self.T-20])
            )
            enhanced, ref1, ref2 = sess.run(
                [step_enhanced, enhanced_ref_1, enhanced_ref_2]
            )
        np.testing.assert_allclose(enhanced[0], ref1)
        np.testing.assert_allclose(enhanced[1, ..., :-20], ref2)

    @retry(5)
    def test_recursive_wpe(self):
        with self.test_session() as sess:
            T = 5000
            D = 2
            K = 1
            delay = 3
            Y = np.random.normal(size=(D, T)) \
                + 1j * np.random.normal(size=(D, T))
            Y = tf.convert_to_tensor(Y[None])
            power = tf.reduce_mean(tf.real(Y) ** 2 + tf.imag(Y) ** 2, axis=1)
            inv_power = tf.reciprocal(power)
            step_enhanced = tf_wpe.wpe_step(
                Y, inv_power, taps=K, delay=D)
            recursive_enhanced = tf_wpe.recursive_wpe(
                tf.transpose(Y, (2, 0, 1)),
                tf.transpose(power),
                1.,
                taps=K,
                delay=D,
                only_use_final_filters=True
            )
            recursive_enhanced = tf.transpose(recursive_enhanced, (1, 2, 0))
            recursive_enhanced, step_enhanced = sess.run(
                [recursive_enhanced, step_enhanced]
            )
        np.testing.assert_allclose(
            recursive_enhanced[..., -200:],
            step_enhanced[..., -200:],
            atol=0.01, rtol=0.2
        )

    def test_batched_recursive_wpe(self):
        with self.test_session() as sess:
            Y_batch, inv_power_batch = self._get_batch_data()
            Y_batch = tf.transpose(Y_batch, (0, 3, 1, 2))
            inv_power_batch = tf.transpose(inv_power_batch, (0, 2, 1))
            enhanced_ref_1 = tf_wpe.recursive_wpe(
                Y_batch[0], inv_power_batch[0], 0.999
            )
            enhanced_ref_2 = tf_wpe.recursive_wpe(
                Y_batch[0, :self.T-20], inv_power_batch[0, :self.T-20],
                0.999
            )
            step_enhanced = tf_wpe.batched_recursive_wpe(
                Y_batch, inv_power_batch, 0.999,
                num_frames=tf.convert_to_tensor([self.T, self.T-20])
            )
            enhanced, ref1, ref2 = sess.run(
                [step_enhanced, enhanced_ref_1, enhanced_ref_2]
            )
        np.testing.assert_allclose(enhanced[0], ref1)
        np.testing.assert_allclose(enhanced[1, :-20], ref2)


if __name__ == '__main__':
    tf.test.main()
