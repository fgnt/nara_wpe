"""
Run all tests with:
    nosetests -w tests/
"""

import unittest
import numpy.testing as tc
import numpy as np
from nara_wpe import wpe
from nara_wpe.test_utils import retry


class TestWPE(unittest.TestCase):
    def setUp(self):
        self.T = np.random.randint(100, 120)
        self.D = np.random.randint(2, 8)
        self.K = np.random.randint(3, 5)
        self.delay = np.random.randint(0, 2)
        self.Y = np.random.normal(size=(self.D, self.T)) \
            + 1j * np.random.normal(size=(self.D, self.T))

    @retry(5)
    def test_filter_operation_v0_vs_v4(self):
        filter_matrix_conj = np.random.normal(size=(self.K, self.D, self.D)) \
            + 1j * np.random.normal(size=(self.K, self.D, self.D))

        desired = wpe.perform_filter_operation_v0(
            self.Y, filter_matrix_conj, self.K, self.delay
        )
        actual = wpe.perform_filter_operation_v4(
            self.Y, filter_matrix_conj, self.K, self.delay
        )
        tc.assert_allclose(desired, actual)

    def test_correlations_narrow_v1_vs_v5(self):
        inverse_power = wpe.get_power_inverse(self.Y)
        R_desired, r_desired = wpe.get_correlations_narrow(
            self.Y, inverse_power, self.K, self.delay
        )
        R_actual, r_actual = wpe.get_correlations_narrow_v5(
            self.Y, inverse_power, self.K, self.delay
        )
        tc.assert_allclose(R_actual, R_desired)
        tc.assert_allclose(r_actual, r_desired)


    @retry(5)
    def test_filter_matrix_conj_v1_vs_v5(self):
        inverse_power = wpe.get_power_inverse(self.Y)

        correlation_matrix, correlation_vector = wpe.get_correlations(
            self.Y, inverse_power, self.K, self.delay
        )
        desired = wpe.get_filter_matrix_conj(
            correlation_matrix, correlation_vector, self.K, self.D
        )
        actual = wpe.get_filter_matrix_conj_v5(
            self.Y, inverse_power, self.K, self.delay
        )
        tc.assert_allclose(actual, desired, atol=1e-6)

    @retry(5)
    def test_filter_matrix_conj_v1_vs_v7(self):
        inverse_power = wpe.get_power_inverse(self.Y)

        correlation_matrix, correlation_vector = wpe.get_correlations(
            self.Y, inverse_power, self.K, self.delay
        )
        desired = wpe.get_filter_matrix_conj(
            correlation_matrix, correlation_vector, self.K, self.D
        )

        s = [Ellipsis, slice(self.delay + self.K - 1, None)]
        Y_tilde = wpe.build_y_tilde(self.Y, self.K, self.delay)
        actual = wpe.get_filter_matrix(
            self.Y, Y_tilde=Y_tilde, inverse_power=inverse_power,
        )
        tc.assert_allclose(
            actual.conj(),
            np.swapaxes(desired, 1, 2).reshape(-1, desired.shape[-1]),
            atol=1e-6
        )

    @retry(5)
    def test_delay_zero_cancels_all(self):
        delay = 0
        X_hat = wpe.wpe(self.Y, self.K, delay=delay)

        # Beginning is never zero. It is a copy of input signal.
        tc.assert_allclose(
            X_hat[:, delay + self.K - 1:],
            np.zeros_like(X_hat[:, delay + self.K - 1:]),
            atol=1e-6
        )

    @retry(5)
    def test_wpe_v0_vs_v7(self):
        desired = wpe.wpe_v0(self.Y, self.K, self.delay, statistics_mode='full')
        actual = wpe.wpe_v7(self.Y, self.K, self.delay, statistics_mode='full')
        tc.assert_allclose(actual, desired, atol=1e-6)

        desired = wpe.wpe_v0(self.Y, self.K, self.delay, statistics_mode='valid')
        actual = wpe.wpe_v7(self.Y, self.K, self.delay, statistics_mode='valid')
        tc.assert_allclose(actual, desired, atol=1e-6)

        desired = wpe.wpe_v6(self.Y, self.K, self.delay, statistics_mode='valid')
        actual = wpe.wpe_v7(self.Y, self.K, self.delay, statistics_mode='full')
        tc.assert_raises(AssertionError, tc.assert_array_equal, desired, actual)

    @retry(5)
    def test_wpe_v8(self):
        desired = wpe.wpe_v6(self.Y, self.K, self.delay, statistics_mode='valid')
        actual = wpe.wpe_v8(self.Y, self.K, self.delay, statistics_mode='valid')
        tc.assert_allclose(actual, desired, atol=1e-6)

        desired = wpe.wpe_v7(self.Y, self.K, self.delay, statistics_mode='valid')
        actual = wpe.wpe_v8(self.Y, self.K, self.delay, statistics_mode='valid')
        tc.assert_allclose(actual, desired, atol=1e-6)

        desired = wpe.wpe_v6(self.Y, self.K, self.delay, statistics_mode='full')
        actual = wpe.wpe_v8(self.Y, self.K, self.delay, statistics_mode='full')
        tc.assert_allclose(actual, desired, atol=1e-6)

        desired = wpe.wpe_v7(self.Y, self.K, self.delay, statistics_mode='full')
        actual = wpe.wpe_v8(self.Y, self.K, self.delay, statistics_mode='full')
        tc.assert_allclose(actual, desired, atol=1e-6)

    @retry(5)
    def test_wpe_multi_freq(self):
        desired = wpe.wpe_v0(self.Y, self.K, self.delay, statistics_mode='full')
        desired = [desired, desired]
        actual = wpe.wpe_v0(np.array([self.Y, self.Y]), self.K, self.delay, statistics_mode='full')
        tc.assert_allclose(actual, desired, atol=1e-6)

        desired = wpe.wpe(self.Y, self.K, self.delay, statistics_mode='full')
        desired = [desired, desired]
        actual = wpe.wpe(np.array([self.Y, self.Y]), self.K, self.delay, statistics_mode='full')
        tc.assert_allclose(actual, desired, atol=1e-6)

    @retry(5)
    def test_wpe_batched_multi_freq(self):
        def to_batched_multi_freq(x):
            return np.array([
                [x, x*2],
                [x*3, x*4],
                [x*5, x*6],
            ])
        Y_batched_multi_freq = to_batched_multi_freq(self.Y)
        print(Y_batched_multi_freq.shape, Y_batched_multi_freq.ndim)

        tc.assert_raises(NotImplementedError, wpe.wpe_v0, Y_batched_multi_freq, self.K, self.delay, statistics_mode='full')

        desired = wpe.wpe_v7(self.Y, self.K, self.delay, statistics_mode='full')
        desired = to_batched_multi_freq(desired)
        actual = wpe.wpe_v7(Y_batched_multi_freq, self.K, self.delay, statistics_mode='full')
        tc.assert_allclose(actual, desired, atol=1e-6)

        desired = wpe.wpe_v8(self.Y, self.K, self.delay, statistics_mode='full')
        desired = to_batched_multi_freq(desired)
        actual = wpe.wpe_v8(Y_batched_multi_freq, self.K, self.delay, statistics_mode='full')
        tc.assert_allclose(actual, desired, atol=1e-6)
