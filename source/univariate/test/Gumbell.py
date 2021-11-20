try:
    from numpy import euler_gamma as _euler_gamma
    import numpy as _np
    from math import sqrt as _sqrt, log as _log, pi as _pi, exp as _exp
    from typing import Union, Dict, List
    from univariate._base import SemiInfinite
except Exception as e:
    print(f"some modules are missing {e}")


class Gumbell(SemiInfinite):
    """
    This class contains methods concerning Gumbel Distirbution [#]_.

    .. math::
        \\text{Gumbel}(x;\\mu,\\beta) = \\frac{1}{\\beta} \\exp{-\\Big( \\frac{x-\\mu}{\\beta} + \\exp{ \\frac{x-\\mu}{\\beta}} \\Big)}

    Args:

        location(float): location parameter (:math:`\\mu`)
        scale(float): scale parameter (:math:`\\beta`) where scale > 0
        x(float): random variable

    Reference:
        .. [#] Wikipedia contributors. (2020, November 26). Gumbel distribution. https://en.wikipedia.org/w/index.php?title=Gumbel_distribution&oldid=990718796
    """

    def __init__(self, location: float, scale: float):
        if scale < 0:
            raise ValueError(
                f'scale parameter should be greater than 0. The value of the scale parameter is: {scale}')

        self.location = location
        self.scale = scale

    def pdf(self, x: Union[List[float], _np.ndarray, float]) -> Union[float, _np.ndarray]:
        """

        Args:
            x (Union[List[float], numpy.ndarray, float]): random variable(s)

        Returns:
            Union[float, numpy.ndarray]: evaluation of pdf at x
        """
        mu = self.location
        beta = self.scale

        if isinstance(x, (_np.ndarray, List)):
            if not type(x) is _np.ndarray:
                x = _np.array(x)
            z = (x-mu)/beta
            return (1/beta)*_np.exp(-(z+_np.exp(-z)))

        z = (x-mu)/beta
        return (1/beta)*_exp(-(z+_exp(-z)))

    def cdf(self, x: Union[List[float], _np.ndarray, float]) -> Union[float, _np.ndarray]:
        """
        Args:
            x (Union[List[float], numpy.ndarray, float]): data point(s) of interest

        Returns:
            Union[float, numpy.ndarray]: evaluation of cdf at x
        """
        mu = self.location
        beta = self.scale

        if isinstance(x, (_np.ndarray, List)):
            if not type(x) is _np.ndarray:
                x = _np.array(x)
            return _np.exp(-_np.exp(-(x-mu)/beta))
        return _exp(-_exp(-(x - mu)/beta))

    def mean(self) -> float:
        """
        Returns: Mean of the Gumbel distribution.
        """
        return self.location+(self.scale*_euler_gamma)

    def median(self) -> float:
        """
        Returns: Median of the Gumbel distribution.
        """
        return self.location - (self.scale*_log(_log(2)))

    def mode(self) -> float:
        """
        Returns: Mode of the Gumbel distribution.
        """
        return self.location

    def var(self) -> float:
        """
        Returns: Variance of the Gumbel distribution.
        """
        return pow(_pi, 2/6)*pow(self.scale, 2)

    def std(self) -> float:
        """
        Returns: Standard deviation of the Gumbel distribution.
        """
        return _sqrt(self.var())

    def skewness(self) -> float:
        """
        Returns: Skewness of the Gumbel distribution.
        """
        return 1.14

    def kurtosis(self) -> float:
        """
        Returns: Kurtosis of the Gumbel distribution.
        """
        return 2.4

    def summary(self) -> Dict[str, float]:
        """
        Returns:
            Dictionary of Gumbel distirbution moments. This includes standard deviation. 
        """
        return {
            'mean': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.var(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }
