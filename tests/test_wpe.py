"""
Run all tests either with:
    nosetests -w tests/
"""

import unittest
import nt.testing as tc
import numpy as np
from nara_wpe import wpe


class TestWPE(unittest.TestCase):
    def test_correlations_v1_vs_v2(self):
        K = 3
        delay = 1
        Y = np.asarray(
            [
                [11, 12, 13, 14],
                [41, 22, 23, 24]
            ], dtype=np.float32
        )

        inverse_power = wpe.get_power_inverse(Y)
        R, r = wpe.get_correlations(Y, inverse_power, K, delay)
        R_v2, r_v2 = wpe.get_correlations_v2(Y, inverse_power, K, delay)

        tc.assert_allclose(R_v2, R)
        tc.assert_allclose(r_v2, r)

    def test_correlations_v1_vs_v2_randomized(self):
        T = np.random.randint(10, 100)
        D = np.random.randint(2, 8)
        K = np.random.randint(3, 5)
        delay = np.random.randint(0, 2)
        Y = np.random.normal(size=(D, T)) + 1j * np.random.normal(size=(D, T))

        inverse_power = wpe.get_power_inverse(Y)
        R, r = wpe.get_correlations(Y, inverse_power, K, delay)
        R_v2, r_v2 = wpe.get_correlations_v2(Y, inverse_power, K, delay)

        tc.assert_allclose(R_v2, R)
        tc.assert_allclose(r_v2, r)

    def test_delay_zero_cancels_all(self):
        """
        If this test fails, it is due to high condition number of
        correlation matrix.
        """
        T = np.random.randint(10, 100)
        D = np.random.randint(2, 8)
        K = np.random.randint(3, 5)
        delay = 0
        Y = np.random.normal(size=(D, T)) + 1j * np.random.normal(size=(D, T))

        X_hat = wpe.wpe(Y, K, delay=delay)

        # Beginning is never zero. Is a copy of input signal.
        tc.assert_allclose(
            X_hat[:, delay + K - 1:],
            np.zeros_like(X_hat[:, delay + K - 1:]),
            atol=1e-10
        )

    def test_filter_operation_v1_vs_v4(self):
        T = np.random.randint(10, 100)
        D = np.random.randint(2, 8)
        K = np.random.randint(3, 5)
        delay = np.random.randint(0, 2)

        Y = np.random.normal(size=(D, T)) + 1j * np.random.normal(size=(D, T))
        filter_matrix_conj = np.random.normal(size=(K, D, D)) \
            + 1j * np.random.normal(size=(K, D, D))

        a = wpe.perform_filter_operation(Y, filter_matrix_conj, K, delay)
        b = wpe.perform_filter_operation_v4(Y, filter_matrix_conj, K, delay)

        tc.assert_allclose(a, b)

    def test_filter_matrix_conj_v1_vs_v3(self):
        """
        If this test fails, it is due to high condition number of
        correlation matrix.
        """
        T = np.random.randint(10, 100)
        D = np.random.randint(2, 8)
        K = np.random.randint(3, 5)
        delay = np.random.randint(0, 2)

        Y = np.random.normal(size=(D, T)) + 1j * np.random.normal(size=(D, T))

        inverse_power = wpe.get_power_inverse(Y)

        correlation_matrix, correlation_vector = wpe.get_correlations(
            Y, inverse_power, K, delay
        )
        ref = wpe.get_filter_matrix_conj(
            correlation_matrix, correlation_vector, K, D
        )
        v3 = wpe.get_filter_matrix_conj_v3(Y, inverse_power, K, delay)

        tc.assert_allclose(v3, ref, atol=1e-10)

    def test_correlations_narrow_v1_vs_v5(self):
        T = np.random.randint(10, 100)
        D = np.random.randint(2, 8)
        K = np.random.randint(3, 5)
        delay = np.random.randint(0, 2)
        Y = np.random.normal(size=(D, T)) + 1j * np.random.normal(size=(D, T))

        inverse_power = wpe.get_power_inverse(Y)
        R, r = wpe.get_correlations_narrow(Y, inverse_power, K, delay)
        R_v5, r_v5 = wpe.get_correlations_narrow_v5(Y, inverse_power, K, delay)

        tc.assert_allclose(R_v5, R)
        tc.assert_allclose(r_v5, r)
