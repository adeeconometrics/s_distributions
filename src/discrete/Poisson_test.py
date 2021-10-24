try: # this is temporary, remove when tests are moved in a separate folder
    import sys
    from os.path import dirname, abspath
    sys.path.append(dirname(dirname(abspath(__file__))))
except Exception as e:
    print(e)

try:
    from unittest import TestCase
    import unittest
    from scipy.special import gammainc
    import numpy as _np
    from math import sqrt, ceil, floor
    from discrete.Poisson import Poisson
    from discrete._base import Base, Finite
except Exception as e:
    print(e)


class TestPoisson(TestCase):

    def setUp(self):
        # test values
        self.l, self.x = 1.5, 5
        self.test_list = [1,2,3,4,5]
        # formulas
        self.pmf = lambda l, k: (pow(l, k) * _np.exp(-l)) / _np.math.factorial(k)
        self.cdf = lambda l,k: gammainc(floor(k + 1), l) / _np.math.factorial(floor(k))
        self.mean = self.l
        self.median = self.l + 1 / 3 - (0.02 / self.l)
        self.mode = ceil(self.l) - 1, floor(self.l)
        self.var = self.l
        self.sk = pow(self.l, -0.5)
        self.ku = 1/self.l

        self.dist = Poisson(self.l)
        
        # test preconditions
        # self.assertIsInstance(self.dist, Infinite)
        self.assertIsInstance(self.dist, Base)
        self.assertTrue(issubclass(Poisson, Base))

        # self.assertRaises(TypeError, Poisson(1.5,2))

    def test_pmf(self):
        self.assertEqual(self.dist.pmf(self.x), self.pmf(self.l, self.x))
        self.assertEqual(self.dist.pmf(self.test_list), [self.pmf(self.l, i) for i in self.test_list])


    def test_cdf(self):
        self.assertEqual(self.dist.cdf(self.x), self.cdf(self.l, self.x))
        self.assertEqual(self.dist.cdf(self.test_list), [self.cdf(self.l, i) for i in self.test_list])

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
