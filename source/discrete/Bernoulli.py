try:
    from math import sqrt as _sqrt
    import numpy as _np
    from typing import Union, Tuple, Dict, List, Literal
    from discrete._base import Finite
except Exception as e:
    print(f"some modules are missing {e}")

class Bernoulli(Finite):
    """
    This class contains methods concerning the Bernoulli Distribution. Bernoulli Distirbution is a special
    case of Binomial Distirbution [#]_ [#]_. 
    
    .. math:: 
        \\text{Bernoulli} (x;p) = p^n (1-p)^{1-x}

    Args:

        - p (float) : event of success. Either 0 or 1. 
        - x (int) : possible outcomes. Either 0 or 1.

    References:
        .. [#] Weisstein, Eric W. "Bernoulli Distribution." From MathWorld--A Wolfram Web Resource. https://mathworld.wolfram.com/BernoulliDistribution.html

        .. [#] Wikipedia contributors. (2020, December 26). Bernoulli distribution. In Wikipedia, The Free Encyclopedia. Retrieved 10:18, December 26, 2020, from https://en.wikipedia.org/w/index.php?title=Bernoulli_distribution&oldid=996380822
    """

    def __init__(self, p:float):
        if p < 0 or p > 1:
            raise ValueError('parameter k is constrained in ∈ [0,1]')
        self.p = p

    def pmf(self, x: Union[List[int], int]) -> Union[int, float, List[Union[int, float]]]:
        """
        Args:

            x (Union[List[int], int]): random variable or list of random variables. Value should either be 0 or 1.

        Returns: 
            probability mass evaluation of Bernoulli distribution to some point specified by the random variable
            or a list of its corresponding value specified by the parameter x.
        """
        p = self.p

        def __generator(p:float, k:int) -> Union[int, float]: 
            if k == 0:
                return 1-p
            return p

        if isinstance(x, List):
            x == _np.array(x)
            if _np.any(x != 0 and x != 1):
                raise ValueError('all x must either be 1 or 0')
            return _np.vectorize(__generator)(p,x)

        if x != 1 and x != 0:
            raise ValueError('all x must either be 1 or 0')
        return __generator(p, x)

    @staticmethod
    def pmf_s(p:float, x: Union[List[int], int]) -> Union[int, float, List[Union[int, float]]]:
        """
        Args:

            x (Union[List[int], int]): random variable or list of random variables. Value should either be 0 or 1.

        Returns: 
            probability mass evaluation of Bernoulli distribution to some point specified by the random variable
            or a list of its corresponding value specified by the parameter x.
        """
        if p < 0 or p > 1:
            raise ValueError('parameter p is constrained in ∈ [0,1]')

        def __generator(p, k) -> Union[int, float]: 
            if k == 0:
                return 1-p
            return p

        if isinstance(x, List):
            x == _np.array(x)
            if _np.any(x != 0 and x != 1):
                raise ValueError('all x must either be 1 or 0')
            return _np.vectorize(__generator)(p,x)

        if x != 1 and x != 0:
            raise ValueError('all x must either be 1 or 0')
        return __generator(p, x)

    def cdf(self, x: Union[List[int], int]) -> Union[int, float, List[Union[int, float]]]:
        """
        Args:

           x (Union[List[int], int]): random variable or list of random variables. Value should either be 0 or 1.

        Returns: 
            commulative density function of Bernoulli distribution to some point specified by the random variable
            or a list of its corresponding value specified by the parameter x.
        """
        p = self.p

        def __generator(p, k) -> Union[int, float]: 
            if k < 0:
                return 0
            elif k >= 0 and k < 1:
                return 1 - p
            else:
                return 1

        if isinstance(x, List):
            x == _np.array(x)
            if _np.any(x != 0 and x != 1):
                raise ValueError('all x must either be 1 or 0')
            return _np.vectorize(__generator)(p,x)

        if x != 1 and x != 0:
            raise ValueError('all x must either be 1 or 0')
        return __generator(p, x)

    def mean(self) -> float:
        """
        Returns: 
            the mean of Bernoulli Distribution.
        """
        return self.p

    def median(self) -> Union[List[int], int]:
        """
        Returns: 
            the median of Bernoulli Distribution.
        """
        p = self.p
        if p < 0.5:
            return 0
        if p == 0.5:
            return [0, 1]
        return 1

    def mode(self) -> Union[Tuple[int, int], int]:
        """
        Returns: 
            the mode of Bernoulli Distribution.
        """
        p = self.p
        if p < 0.5:
            return 0
        if p == 0.5:
            return (0, 1)
        return 1

    def var(self) -> float:
        """
        Returns: 
            the variance of Bernoulli Distribution.
        """
        p = self.p
        q = 1 - p
        return p * q

    def std(self) -> float:
        """
        Returns: 
            the variance of Bernoulli Distribution.
        """
        p = self.p
        q = 1 - p
        return _sqrt(p * q)

    def skewness(self) -> float:
        """
        Returns: 
            the skewness of Bernoulli Distribution.
        """
        p = self.p
        q = 1 - p
        return (q - p) / _sqrt(p * q)

    def kurtosis(self) -> float:
        """
        Returns: 
            the kurtosis of Bernoulli Distribution.
        """
        p = self.p
        q = 1 - p
        return (1 - 6 * p * q) / (p * q)

    def summary(self) -> Dict[str, Union[int, float, List[int], Tuple[int, int]]]:
        """
        Summary statistic regarding the Bernoulli distribution which contains the following parts of the distribution:
                (mean, median, mode, var, std, skewness, kurtosis).

        Returns: Dict[str, Union[int, List[int], Tuple[int, int]]]
        """
        return {
            'mean': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.var(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }