try: # this is temporary, remove when tests are moved in a separate folder
    import sys
    from os.path import dirname, abspath
    sys.path.append(dirname(dirname(abspath(__file__))))
except Exception as e:
    print(e)

try:
    from unittest import TestCase
    import unittest
    from scipy.special import binom
    import numpy as _np
    from math import sqrt, ceil, floor
    from discrete.Hypergeometric import Hypergeometric
    from discrete._base import Base, Finite
except Exception as e:
    print(e)


class TestHypergeometric(TestCase):

    def setUp(self):
        # test values
        N,K,k,n = 50, 5, 4, 10
        self.N, self.K, self.k, self.n = N,K,k,n
        self.test_list = [1,2,3,4,5]
        # formulas
        self.pmf = lambda N,K,k,n: binom(K,k)*binom(N-K, n-k)/binom(N,n)
        # self.cdf = lambda p,k: 0 if k < 0 else (1-p if k >= 0 and k <1 else 1)
        self.mean = self.n*self.K/self.N
        self.median = "undefined"
        self.mode = ceil((n+1)*(K+1)/(N+2))-1, floor((n+1)*(K+1)/(N+2))
        self.var = n*(K/N)*(N - K)/N * (N - n)/(N-1)
        self.sk = ((N - 2 * K) * sqrt(N - 1) *(N - 2 * n)) / (sqrt(n * K * (N - K) * (N - n)) * (N - 2))
        scale = 1 / (n * k*(N - K) * (N - n) * (N - 2) * (N - 3))
        self.ku = scale * ((N - 1) * N**2 * (N * (N + 1) - (6 * K * (N - K)) -
                                          (6 * n * (N - n))) +
                        (6 * n * K*(N - K) * (N - n) * (5 * N - 6)))

        self.dist = Hypergeometric(N,K,k,n)
        
        # test preconditions
        self.assertIsInstance(self.dist, Finite)
        self.assertIsInstance(self.dist, Base)
        self.assertTrue(issubclass(Hypergeometric, Base))

        # self.assertRaises(TypeError, Hypergeometric(1.5,2))

    def test_pmf(self):
        self.assertEqual(self.dist.pmf(), self.pmf(self.N, self.K, self.k, self.n))

    # def test_cdf(self):
    #     self.assertEqual(self.dist.cdf(self.k), self.cdf(self.p,1))
    #     self.assertEqual(self.dist.cdf(self.test_list), [self.cdf(self.p, i) for i in self.test_list])

    def test_mean(self):
        self.assertEqual(self.dist.mean(), self.mean)

    def test_median(self):
        self.assertEqual(self.dist.median(), self.median)

    def test_mode(self):
        self.assertEqual(self.dist.mode(), self.mode)

    def test_var(self):
        self.assertEqual(self.dist.var(), self.var)

    def test_std(self):
        self.assertEqual(self.dist.std(), sqrt(self.var))

    def test_skewness(self):
        self.assertEqual(self.dist.skewness(), self.sk)

    def test_kurtosis(self):
        self.assertEqual(self.dist.kurtosis(), self.ku)

    def test_summary(self):
        pass

    def test_keys(self):
        pass
