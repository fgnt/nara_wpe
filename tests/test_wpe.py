"""
Run all tests either with:
    nosetests -w tests/
"""

import unittest
import nt.testing as tc
import numpy as np
from quarry import wpe


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
