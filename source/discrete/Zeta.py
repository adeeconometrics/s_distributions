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
    This class contains methods concerning the Zeta Distribution [#]_ [#]_.

    .. math:: \\text{Zeta}(x;s) =\\frac{\\frac{1}{x^s}}{\zeta(s)}

    Args:
        - s (float): main parameter
        - x (int): support parameter

    References:
        .. [#] Wikipedia contributors. (2020, November 6). Zeta distribution. In Wikipedia, The Free Encyclopedia. Retrieved 10:24, December 26, 2020, from https://en.wikipedia.org/w/index.php?title=Zeta_distribution&oldid=987351423
        .. [#] The Zeta Distribution. (2021, February 3). https://stats.libretexts.org/@go/page/10473
    """

    def __init__(self, s: float):
        self.s = s

    def pmf(self, x: Union[List[int], int]) -> Union[int, float, List[int]]:
        """
        Args:
            x (Union[List[int], int]): random variables

        Returns:
            Union[int, float, List[int]]: evaluation of pmf at x
        """
        s = self.s
        def __generator(s, k): return (1 / k**s) / _zeta(s)

        if isinstance(x, List):
            return [__generator(s, i) for i in x]  # double check this function

        return __generator(s, x)

    def cdf(self, x: List[int] = None) -> Union[int, float, List[int]]:
        """
        Args:
            x (List[int], optional): random variables. Defaults to None.

        Returns:
            Union[int, float, List[int]]: evaluation of cdf at x. Currently NotImplemented
        """
        return NotImplemented

    def mean(self) -> Union[str, float]:
        """
        Returns: 
            mean of Zeta distribution
        """
        s = self.s
        if s > 2:
            return _zeta(s - 1) / _zeta(s)
        return "undefined"

    def median(self) -> str:
        """
        Returns: 
            undefined.
        """
        return "undefined"

    def mode(self) -> int:
        """
        Returns: 
            mode of Zeta distribution
        """
        return 1

    def var(self) -> Union[str, float]:
        """
        Returns: 
            the variance of Zeta Distribution. Returns undefined if s <= 3.
        """
        s = self.s
        if s > 3:
            _x0 = _zeta(s)
            return (_zeta(s-2)/_x0) - (_zeta(s-1)/_zeta(s))**2
        return "undefined"

    def std(self) -> Union[str, float]:
        """
        Returns:
            the standard deviation of Zeta Distribution. Returns undefined if variance is undefined.
        """
        s = self.s
        if s > 3:
            _x0 = _zeta(s)
            return _sqrt((_zeta(s-2)/_x0) - (_zeta(s-1)/_zeta(s))**2)
        return "undefined"

    def skewness(self) -> Union[str, float]:
        """
        Returns: 
            the skewness of Zeta Distribution.
        """
        s = self.s
        if s <= 4:
            return "undefined"
        _x0 = _zeta(s-2)*_zeta(s)
        _x1 = _zeta(s-1)
        return (_zeta(s-3)*_zeta(s)**2 - 3*_x1*_x0 + 2*_x1**3) / pow(_x0-_x1**2, 3/2)

    def kurtosis(self) -> Union[str, float]:
        """
        Returns: 
            the kurtosis of Zeta Distribution.
        """
        s = self.s 
        if s <= 5:
            return "undefined"

        _x0 = _zeta(s-2)
        _x1 = _zeta(s)
        _x3 = _zeta(s-1)

        scale = 1/pow(_x0*_x1 - _x3**2,2)
        numerator = (_zeta(s-4)*_x1**3) - (4*_x3*_zeta(s-3)*_x1**2) + (6*_x3**2*_x0*_x1-3*_x3**4)
        return scale*numerator

    def summary(self) -> Dict[str, Union[float, int, str]]:
        """
        Returns:
            Dictionary of Zeta distirbution moments. This includes standard deviation. 
        """
        return {
            'mean': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.var(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }
