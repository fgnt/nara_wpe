"""
Run all tests with:
    nosetests -w tests/
"""

import unittest
import nt.testing as tc
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

    def test_correlations_v1_vs_v2_toy_example(self):
        K = 3
        delay = 1
        Y = np.asarray(
            [
                [11, 12, 13, 14],
                [41, 22, 23, 24]
            ], dtype=np.float32
        )
        inverse_power = wpe.get_power_inverse(Y)
        R_desired, r_desired = wpe.get_correlations(Y, inverse_power, K, delay)
        R_actual, r_actual = wpe.get_correlations_v2(Y, inverse_power, K, delay)
        tc.assert_allclose(R_actual, R_desired)
        tc.assert_allclose(r_actual, r_desired)

    def test_correlations_v1_vs_v2(self):
        inverse_power = wpe.get_power_inverse(self.Y)
        R_desired, r_desired = wpe.get_correlations(
            self.Y, inverse_power, self.K, self.delay
        )
        R_actual, r_actual = wpe.get_correlations_v2(
            self.Y, inverse_power, self.K, self.delay
        )
        tc.assert_allclose(R_actual, R_desired)
        tc.assert_allclose(r_actual, r_desired)

    @retry(5)
    def test_wpe_v1_vs_v2(self):
        desired = wpe.wpe_v1(self.Y, self.K, self.delay)
        actual = wpe.wpe_v2(self.Y, self.K, self.delay)
        tc.assert_allclose(actual, desired)

    @retry(5)
    def test_filter_operation_v1_vs_v4(self):
        filter_matrix_conj = np.random.normal(size=(self.K, self.D, self.D)) \
            + 1j * np.random.normal(size=(self.K, self.D, self.D))

        desired = wpe.perform_filter_operation(
            self.Y, filter_matrix_conj, self.K, self.delay
        )
        actual = wpe.perform_filter_operation_v4(
            self.Y, filter_matrix_conj, self.K, self.delay
        )
        tc.assert_allclose(desired, actual)

    @retry(5)
    def test_wpe_v1_vs_v4(self):
        desired = wpe.wpe_v1(self.Y, self.K, self.delay)
        actual = wpe.wpe_v4(self.Y, self.K, self.delay)
        tc.assert_allclose(actual, desired)

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
        tc.assert_allclose(actual, desired, atol=1e-10)

    @retry(5)
    def test_wpe_v1_vs_v5(self):
        desired = wpe.wpe_v1(self.Y, self.K, self.delay)
        actual = wpe.wpe_v5(self.Y, self.K, self.delay)
        tc.assert_allclose(actual, desired)

    @retry(5)
    def test_delay_zero_cancels_all(self):
        delay = 0
        X_hat = wpe.wpe(self.Y, self.K, delay=delay)

        # Beginning is never zero. It is a copy of input signal.
        tc.assert_allclose(
            X_hat[:, delay + self.K - 1:],
            np.zeros_like(X_hat[:, delay + self.K - 1:]),
            atol=1e-10
        )
