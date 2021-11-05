try: # this is temporary, remove when tests are moved in a separate folder
    import sys
    from os.path import dirname, abspath
    sys.path.append(dirname(dirname(abspath(__file__))))
except Exception as e:
    print(e)

try:
    from unittest import TestCase
    from math import sqrt
    from Binomial import Binomial
    from _base import Base
except Exception as e:
    print(e)


class TestBinomial(TestCase):

    def setUp(self):
        # values
        self.p, self.k = 0.5, 1
        self.test_list = [1,0,1,0,1,0,0,1]
        # formula
        self.pmf = lambda p,k: 1-p if k == 0 else p
        self.cdf = lambda p,k: 0 if k < 0 else (1-p if k >= 0 and k <1 else 1)
        self.mean = self.p
        self.median = lambda p: 0 if p < 0.5 else ((0,1) if p == 0.5 else 1) 
        self.mode = lambda p: 0 if p < 0.5 else ((0,1) if p == 0.5 else 1) 
        self.var = lambda p: (1-p)*p
        self.sk = lambda p: (1-2*p)/sqrt((1-p)*p)
        self.ku = lambda p: 1/self.var(p) - 6 # got from Distributions.jl

        self.dist = Binomial(self.p)
        
        # test preconditions
        self.assertIsInstance(self.dist, Base)
        # self.assertRaises(TypeError, Binomial(1.5,2))

    def test_pmf(self):
        self.assertEqual(self.dist.pmf(self.k), self.pmf(self.p,1))
        self.assertEqual(self.dist.pmf(self.test_list), [self.pmf(self.p, i) for i in self.test_list])


    def test_cdf(self):
        self.assertEqual(self.dist.cdf(self.k), self.cdf(self.p,1))
        self.assertEqual(self.dist.cdf(self.test_list), [self.cdf(self.p, i) for i in self.test_list])

    def test_mean(self):
        self.assertEqual(self.dist.mean(), self.p)

    def test_median(self):
        self.assertEqual(self.dist.median(), self.median(self.p))

    def test_mode(self):
        self.assertEqual(self.dist.mode(), self.mode(self.p))

    def test_var(self):
        self.assertEqual(self.dist.var(), self.var(self.p))

    def test_std(self):
        self.assertEqual(self.dist.std(), sqrt(self.var(self.p)))

    def test_skewness(self):
        self.assertEqual(self.dist.skewness(), self.sk(self.p))

    def test_kurtosis(self):
        self.assertEqual(self.dist.kurtosis(), self.ku(self.p))

    def test_summary(self):
        pass

    def test_keys(self):
        pass
