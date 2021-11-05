try: # this is temporary, remove when tests are moved in a separate folder
    import sys
    from os.path import dirname, abspath
    sys.path.append(dirname(dirname(abspath(__file__))))
except Exception as e:
    print(e)

try:
    from unittest import TestCase
    import unittest
    from math import sqrt, log2, ceil
    from Geometric import Geometric
    from _base import Base, Finite
except Exception as e:
    print(e)


class TestGeometric(TestCase):

    def setUp(self):
        # values
        self.p, self.k = 0.5, 1
        self.test_list = [1,1,2,3,4,5,6]
        # formula
        self.pmf_t1 = lambda p,k: pow(1-p, k-1)*p
        self.pmf_t2 = lambda p,k: pow(1-p,k)*p
        self.cdf_t1 = lambda p,k: 1-pow(1-p,k)
        self.cdf_t2 = lambda p,k: 1-pow(1-p, k+1)
        self.mean_t1 = 1/self.p
        self.mean_t2 = (1-self.p)/self.p
        self.median_t1 = ceil(-1/log2(1-self.p))
        self.median_t2 = ceil(-1/log2(1-self.p)) - 1
        self.mode_t1 = 1
        self.mode_t2 = 0
        self.var = (1-self.p)/self.p**2
        self.sk = (2-self.p)/sqrt(1-self.p)
        self.ku = 6 + self.p**2/(1-self.p)

        self.dist = Geometric(self.p)
        
        # test preconditions
        self.assertIsInstance(self.dist, Base)
        self.assertIsInstance(self.dist, Finite)
        self.assertIsInstance(self.dist, Base)
        self.assertTrue(issubclass(Geometric, Base))
        # self.assertRaises(TypeError, Geometric(1.5,2))

    def test_pmf(self):
        self.assertEqual(self.dist.pmf(self.k), self.pmf_t1(self.p,1))
        self.assertEqual(self.dist.pmf(self.test_list), [self.pmf_t1(self.p, i) for i in self.test_list])

        self.assertEqual(self.dist.pmf(self.k, _type='second'), self.pmf_t2(self.p,1))
        self.assertEqual(self.dist.pmf(self.test_list, _type='second'), [self.pmf_t2(self.p, i) for i in self.test_list])

    def test_cdf(self):
        self.assertEqual(self.dist.cdf(self.k), self.cdf_t1(self.p,1))
        self.assertEqual(self.dist.cdf(self.test_list), [self.cdf_t1(self.p, i) for i in self.test_list])

        self.assertEqual(self.dist.cdf(self.k, _type='second'), self.cdf_t2(self.p,1))
        self.assertEqual(self.dist.cdf(self.test_list, _type='second'), [self.cdf_t2(self.p, i) for i in self.test_list])

    def test_mean(self):
        self.assertEqual(self.dist.mean(), self.mean_t1)
        self.assertEqual(self.dist.mean(_type='second'), self.mean_t2)

    def test_median(self):
        self.assertEqual(self.dist.median(), self.median_t1)
        self.assertEqual(self.dist.median(_type='second'), self.median_t2)

    def test_mode(self):
        self.assertEqual(self.dist.mode(), self.mode_t1)
        self.assertEqual(self.dist.mode(_type='second'), self.mode_t2)

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
