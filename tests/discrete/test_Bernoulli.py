from unittest import TestCase
import unittest
from math import sqrt
import src
from src.discrete._base import Base
from src.discrete.Bernoulli import Bernoulli


class TestBernoulli(TestCase):

    def setUp(self):
        self.bernoulli = Bernoulli(10, 1.0)

    def tearDown(self):
        pass

    def test_pmf(self):
        # self.assertIsInstance(self.bernoulli, Base)
        self.assertEqual(self.bernoulli.pmf(), 10**1 * 1)

    def test_cdf(self):
        pass

    def test_mean(self):
        pass

    def test_median(self):
        pass

    def test_mode(self):
        pass

    def test_var(self):
        pass

    def test_std(self):
        pass

    def test_skewness(self):
        pass

    def test_kurtosis(self):
        pass

    def test_summary(self):
        pass

    def test_keys(self):
        pass
