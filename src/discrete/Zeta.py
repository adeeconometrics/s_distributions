try:
    from scipy.special import zeta as _zeta
    import numpy as np
    from math import sqrt as _sqrt, ceil as _ceil, floor as _floor, log2 as _log2
    from typing import Union, Tuple, Dict, List
    from discrete._base import Base
except Exception as e:
    print(f"some modules are missing {e}")


class Zeta(Base):
    """
    This class contains methods concerning the Zeta Distribution.

    Args:
        - s(float): main parameter
        - k(int): support parameter

    References:
        - Wikipedia contributors. (2020, November 6). Zeta distribution. In Wikipedia, The Free Encyclopedia. 
        Retrieved 10:24, December 26, 2020, from https://en.wikipedia.org/w/index.php?title=Zeta_distribution&oldid=987351423
    """

    def __init__(self, s: Union[int, float], k: int):
        if type(k) is not int:
            raise TypeError('parameter k must be of type int')

        self.s = s
        self.k = k

    def pmf(self, x: List[int] = None) -> Union[int, float, List[int]]:
        """
        Args:

            x (List[int]): random variable or list of random variables

        Returns: 
            either probability mass evaluation for some point or scatter plot of Zeta distribution.
        """
        s = self.s
        k = self.k
        def __generator(s, k): return (1 / k**6) / _zeta(s)

        if x is not None and issubclass(x, List):
            return [__generator(s, i) for i in x]  # double check this function

        return __generator(s, k)

    def cdf(self, x: List[int] = None) -> Union[int, float, List[int]]:
        """
        Args:

            x (List[int]): random variable or list of random variables

        Returns: 
            either cumulative distribution evaluation for some point or scatter plot of Zeta distribution.
        """
        pass

    def mean(self) -> Union[str, float]:
        """
        Returns the mean of Zeta Distribution. Returns None if undefined.
        """
        s = self.s
        if s > 2:
            return _zeta(s - 1) / _zeta(s)
        return "undefined"

    def median(self) -> str:
        """
        Returns the median of Zeta Distribution. Retruns None if undefined.
        """
        return "undefined"

    def mode(self) -> int:
        """
        Returns the mode of Zeta Distribution.
        """
        return 1

    def var(self) -> Union[str, float]:
        """
        Returns the variance of Zeta Distribution. Returns None if undefined.
        """
        s = self.s
        if s > 3:
            return (_zeta(s) * _zeta(s - 1) - _zeta(s - 1)**2) * 1/_zeta(s)**2
        return "undefined"

    def skewness(self) -> str:
        """
        Returns the skewness of Zeta Distribution. Currently unsupported.
        """
        return "unsupported"

    def kurtosis(self) -> str:
        """
        Returns the kurtosis of Zeta Distribution. Currently unsupported.
        """
        return "unsupported"

    def summary(self) -> Dict[str, Union[float, int]]:
        """
        Summary statistic regarding the Zeta distribution which contains the following parts of the distribution:
        (mean, median, mode, var, std, skewness, kurtosis).

        Returns:
            Dict[str, Union[float, int]]
        """
        return {
            'mean': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.var(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }
