try:
    from scipy.special import binom as _binom
    import numpy as _np
    from typing import Union, Tuple, Dict, List
    from math import sqrt as _sqrt, factorial as _factorial
    from univariate._base import BoundedInterval
except Exception as e:
    print(f"some modules are missing {e}")


class Bates(BoundedInterval):
    """
    This class contains methods concerning Bates Distirbution. Also referred to as the regular mean distribution.

    Note that the Bates distribution is a probability distribution of the mean of a number of statistically indipendent uniformly
    distirbuted random variables on the unit interval. This is often confused with the Irwin-Hall distribution which is
    the distribution of the sum (not the mean) of n independent random variables. The two distributions are simply versions of
    each other as they only differ in scale [#]_.

    
    Args:

        a(float): lower bound parameter 
        b(float): upper bound parameter where b > a
        n(int): where n >= 1 
        randvar(float | [a,b]): random variable

    Reference:
        .. [#] Wikipedia contributors. (2021, January 8). Bates distribution. https://en.wikipedia.org/w/index.php?title=Bates_distribution&oldid=999042206
    """

    def __init__(self, a: float, b: float, n: int, randvar: float):
        if randvar < 0 or randvar > 1:
            raise ValueError(
                f'random variable should only be in between 0 and 1. Entered value: {randvar}')
        if a > b:
            raise ValueError(
                'lower bound (a) should not be greater than upper bound (b).')
        if type(n) is not int:
            raise TypeError('parameter n should be an integer type.')

        self.a = a
        self.b = b
        self.n = n
        self.randvar = randvar

    def cdf(self, x: Union[List[float], _np.ndarray] = None) -> Union[float, _np.ndarray]:
        """
        Args:

            x (List[float], numpy.ndarray): random variable or list of random variables

        Returns:
            either cumulative distribution evaluation for some point or plot of Bates distribution.
        """
        return "currently unsupported"

    def mean(self) -> float:
        """
        Returns: Mean of the Bates distribution.
        """
        return 0.5*(self.a+self.b)

    def var(self) -> float:
        """
        Returns: Variance of the Bates distribution.
        """
        return 1/(12*self.n)*pow(self.b-self.a, 2)

    def std(self) -> float:
        """
        Returns: Standard devtiation of the Bates distribution
        """
        return _sqrt(1/(12*self.n)*pow(self.b-self.a, 2))

    def skewness(self) -> float:
        """
        Returns: Skewness of the Bates distribution.
        """
        return -6/(5*self.n)

    def kurtosis(self) -> float:
        """
        Returns: Kurtosis of the Bates distribution.
        """
        return 0.0

    def summary(self) -> Dict[str, Union[float, Tuple[float]]]:
        """
        Summary statistic regarding the Bates distribution which contains the following parts of the distribution:
        (mean, median, mode, var, std, skewness, kurtosis).

        Returns:
            Dict[str, Union[float, Tuple[float]]]: [description]
        """
        return {
            'mean': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.var(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }
