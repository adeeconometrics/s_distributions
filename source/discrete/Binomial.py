try:
    from scipy.special import binom as _binom, betainc as _betainc
    import numpy as _np
    from math import sqrt as _sqrt, ceil as _ceil, floor as _floor
    from typing import Union, Tuple, Dict, List
    from _base import Finite
except Exception as e:
    print(f"some modules are missing {e}")


class Binomial(Finite):
    """
    This class contains functions for finding the probability mass function and 
    cumulative distribution function for binomial distirbution [#]_ [#]_ [#]_. 

    .. math::
        \\text{Binomial}(x;n,p) = \\binom{n}{x} p^k (1-p)^{n-x}

    Args:

        n (int): number  of trials
        p (float): success probability for each trial. Where 0 <= p <= 1.
        x (int): number of successes 


    References:
        .. [#] NIST/SEMATECH e-Handbook of Statistical Methods (2012). Binomial Distribution. Retrieved at http://www.itl.nist.gov/div898/handbook/, December 26, 2000.
        .. [#] Wikipedia contributors. (2020, December 19). Binomial distribution. In Wikipedia, The Free Encyclopedia. Retrieved 07:24, December 26, 2020, from https://en.wikipedia.org/w/index.php?title=Binomial_distribution&oldid=995095096
        .. [#] Weisstein, Eric W. "Binomial Distribution." From MathWorld--A Wolfram Web Resource. https://mathworld.wolfram.com/BinomialDistribution.html
    """

    def __init__(self, n: int, p: float):
        if p < 0 or p > 1:
            raise ValueError('parameter p is constrained to ∈ [0,1]')

        self.n = n
        self.p = p

    def pmf(self, x: Union[List[int], int, _np.ndarray]) -> Union[int, _np.ndarray]:
        """
        Args:
            x (Union[List[int], int]): random variable or list of random variables

        Returns:
            Union[int, numpy.ndarray]: evaluation of pmf at x
        """
        n = self.n
        p = self.p

        if isinstance(x, (List,_np.ndarray)):
            if not type(x) is _np.ndarray:
                x = _np.array(x)
        return _binom(n,x)*p**x*(1-p)**(n-x)

    def cdf(self, x:Union[int, List[int], _np.ndarray]) -> Union[int, _np.ndarray]:
        """
        Args:
            x (Union[int, List[int], _np.ndarray]): random variable or list of random variables

        Returns:
            Union[int, numpy.ndarray]: evaluation of cdf at x
        """

        n = self.n
        p = self.p

        if isinstance(x,List):
            if not type(x) is _np.ndarray:
                x = _np.array(x)
        return _betainc(n-x,1+x,1-p)

    def mean(self) -> float:
        """
        Returns: 
            the mean of Binomial Distribution.
        """
        return self.n * self.p

    def median(self) -> Tuple[int, int]:
        """
        Returns: 
            the median of Binomial Distribution. Either one defined in the tuple of result.
        """
        n = self.n
        p = self.p
        return _floor(n * p), _ceil(n * p)

    def mode(self) -> Tuple[int, int]:
        """
        Returns: 
            the mode of Binomial Distribution. Either one defined in the tuple of result.
        """
        n = self.n
        p = self.p
        return _floor((n + 1) * p), _ceil((n + 1) * p) - 1

    def var(self) -> float:
        """
        Returns: 
            the variance of Binomial Distribution.
        """
        n = self.n
        p = self.p
        q = 1 - p
        return n * p * q

    def skewness(self) -> float:
        """
        Returns: 
            the skewness of Binomial Distribution.
        """
        n = self.n
        p = self.p
        q = 1 - p
        return (q - p) / _sqrt(n * p * q)

    def kurtosis(self) -> float:
        """
        Returns: 
            the kurtosis of Binomial Distribution.
        """
        n = self.n
        p = self.p
        q = 1 - p
        return (1 - 6 * p * q) / (n * p * q)

    def keys(self) -> Dict[str, Union[float, int, Tuple[int, int]]]:
        """
        Returns:
            Dictionary of Binomial distirbution moments. This includes standard deviation. 
        """
        return {
            'mean': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.var(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }
