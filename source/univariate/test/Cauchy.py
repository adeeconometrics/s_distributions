try:
    import numpy as _np
    from math import log as _log, log10 as _log10, pi as _pi, atan as _atan
    from typing import Union, Dict, List
    from univariate._base import Infinite
except Exception as e:
    print(f"some modules are missing {e}")


class Cauchy(Infinite):
    """
    This class contains methods concerning the Cauchy Distribution [#]_ [#]_.

    .. math::
        \\text{Cauchy}(x;loc, scale) = \\frac{1}{\\pi \cdot scale \\big[ 1 + \\big( \\frac{x-loc}{scale} \\big)^2 \\big]}

    Args:

        loc(float): pertains to the loc parameter or median
        scale(float): pertains to  the scale parameter where scale > 0
        x(float): random variable

    References:
        .. [#] Wikipedia contributors. (2020, November 29). Cauchy distribution. https://en.wikipedia.org/w/index.php?title=Cauchy_distribution&oldid=991234690
        .. [#] Weisstein, Eric W. "Cauchy Distribution." From MathWorld--A Wolfram Web Resource. https://mathworld.wolfram.com/CauchyDistribution.html
    """

    def __init__(self, loc: float, scale: float):
        if scale < 0:
            raise ValueError('scale should be a positive number.')
        self.scale = scale
        self.loc = loc

    def pdf(self, x: Union[List[float], _np.ndarray, float]) -> Union[float, _np.ndarray]:
        """
        Args:
            x (Union[List[float], numpy.ndarray, float]): random variable(s)

        Returns:
            Union[float, numpy.ndarray]: evaluation of pdf at x
        """
        loc = self.loc
        scale = self.scale

        if isinstance(x, (_np.ndarray, List)):
            if not type(x) is _np.ndarray:
                x = _np.array(x)
            return 1/(_pi * scale * (1 + _np.power((x - loc) / scale, 2)))

        return 1/(_pi * scale * (1 + pow((x - loc) / scale, 2)))

    def cdf(self, x: Union[List[float], _np.ndarray, float]) -> Union[float, _np.ndarray]:
        """
        Args:
            x (Union[List[float], numpy.ndarray, float]): data point(s) of interest

        Returns:
            Union[float, numpy.ndarray]: evaluation of cdf at x
        """
        loc = self.loc
        scale = self.scale

        if isinstance(x, (_np.ndarray, List)):
            if not type(x) is _np.ndarray:
                x = _np.array(x)
            return (1 / _pi) * _np.arctan((x - loc) / scale) + 0.5

        return (1 / _pi) * _atan((x - loc) / scale) + 0.5

    def mean(self) -> str:
        """
        Returns: Mean of the Cauchy distribution. Mean is Undefined.
        """
        return "undefined"

    def median(self) -> float:
        """
        Returns: Median of the Cauchy distribution.
        """
        return self.loc

    def mode(self) -> float:
        """
        Returns: Mode of the Cauchy distribution
        """
        return self.loc

    def var(self) -> str:
        """
        Returns: Variance of the Cauchy distribution.
        """
        return "undefined"

    def std(self) -> str:
        """
        Returns: Standard Deviation of the Cauchy Distribution.
        """
        return "undefined"

    def skewness(self) -> str:
        """
        Returns: Skewness of the Cauchy distribution.
        """
        return "undefined"

    def kurtosis(self) -> float:
        """
        Returns: Kurtosis of the Cauchy distribution
        """
        return _log(4 * _pi * self.scale)

    def entropy(self) -> float:
        """
        Returns: differential entropy of the Cauchy distribution

        Reference: Park, S.Y. & Bera, A.K.(2009). Maximum entropy autoregressive conditional heteroskedasticity model. Elsivier.
        link: http://wise.xmu.edu.cn/uploadfiles/paper-masterdownload/2009519932327055475115776.pdf
        """
        return _log10(4*_pi*self.scale)

    def summary(self) -> Dict[str, Union[float, str]]:
        """
        Returns:
            Dictionary of Cauchy distirbution moments. This includes standard deviation. 
        """
        return {
            'mean': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.var(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }
