try:
    import numpy as _np
    from numpy import euler_gamma as _euler_gamma
    from typing import Union, Tuple, Dict, List
    from math import sqrt as _sqrt, log as _log, pi as _pi, exp as _exp
    from univariate._base import SemiInfinite
except Exception as e:
    print(f"some modules are missing {e}")

# change x to x


class Rayleigh(SemiInfinite):
    """
    This class contains methods concerning Rayleigh Distirbution [#]_ [#]_.

    .. math:: \\text{Rayleigh}(x;\\sigma) = \\frac{x}{\\sigma^2} \\exp{-(x^2/(2\\sigma^2))}

    Args:

        scale(float): scale parameter (:math:`\\sigma`) where scale > 0
        x(float): random variable where x >= 0

    Reference:
        .. [#] Wikipedia contributors. (2020, December 30). Rayleigh distribution. https://en.wikipedia.org/w/index.php?title=Rayleigh_distribution&oldid=997166230
        .. [#] Weisstein, Eric W. "Rayleigh Distribution." From MathWorld--A Wolfram Web Resource. https://mathworld.wolfram.com/RayleighDistribution.html
    """

    def __init__(self, scale: float):
        if scale < 0:
            raise ValueError('scale parameter should be a positive number.')

        self.scale = scale

    def pdf(self, x: Union[List[float], _np.ndarray, float]) -> Union[float, _np.ndarray]:
        """
        Args:
            x (Union[List[float], numpy.ndarray, float]): random variable(s)

        Raises:
            ValueError: when there exist a value of x that is less than 0

        Returns:
            Union[float, numpy.ndarray]: evaluation of pdf at x
        """
        sig = self.scale  # scale to sig

        if isinstance(x, (_np.ndarray, List)):
            if not type(x) is _np.ndarray:
                x = _np.array(x)
            if _np.any(x < 0):
                raise ValueError('random variable must be a positive number')
            return x/pow(sig, 2) * _np.exp(_np.power(-x, 2)/(2*pow(sig, 2)))

        if x < 0:
            raise ValueError('random variable must be a positive number')
        return x/pow(sig, 2) * _exp(pow(-x, 2)/(2*pow(sig, 2)))

    def cdf(self, x: Union[List[float], _np.ndarray, float]) -> Union[float, _np.ndarray]:
        """
        Args:
            x (Union[List[float], numpy.ndarray, float]): data point(s) of interest

        Returns:
            Union[float, numpy.ndarray]: evaluation of cdf at x
        """
        sig = self.scale

        if isinstance(x, (_np.ndarray, List)):
            if not type(x) is _np.ndarray:
                x = _np.array(x)
            return 1-_np.exp(-_np.power(x, 2)/(2*sig**2))

        return 1-_exp(-x**2/(2*sig**2))

    def mean(self) -> float:
        """
        Returns: Mean of the Rayleigh distribution.
        """
        return self.scale*_sqrt(_pi/2)

    def median(self) -> float:
        """
        Returns: Median of the Rayleigh distribution.
        """
        return self.scale*_sqrt(2*_log(2))

    def mode(self) -> float:
        """
        Returns: Mode of the Rayleigh distribution.
        """
        return self.scale

    def var(self) -> float:
        """
        Returns: Variance of the Rayleigh distribution.
        """
        return (4-_pi)/2*pow(self.scale, 2)

    def std(self) -> float:
        """
        Returns: Standard deviation of the Rayleigh distribution
        """
        return _sqrt((4-_pi)/2*pow(self.scale, 2))

    def skewness(self) -> float:
        """
        Returns: Skewness of the Rayleigh distribution.
        """
        return (2*_sqrt(_pi)*(_pi-3))/pow((4-_pi), 3/2)

    def kurtosis(self) -> float:
        """
        Returns: Kurtosis of the Rayleigh distribution.
        """
        return -(6*pow(_pi, 2)-24*_pi+16)/pow(4-_pi, *2)

    def entropy(self) -> float:
        """
        Returns: differential entropy of the Rayleigh distribution.

        Reference: Park, S.Y. & Bera, A.K.(2009). Maximum entropy autoregressive conditional heteroskedasticity model. Elsivier.
        link: http://wise.xmu.edu.cn/uploadfiles/paper-masterdownload/2009519932327055475115776.pdf
        """
        return 1+_log(self.scale/_sqrt(2))+(_euler_gamma/2)

    def summary(self) -> Dict[str, Union[float, Tuple[float]]]:
        """
        Returns:
            Dictionary of Rayleigh distirbution moments. This includes standard deviation. 
        """
        return {
            'mean': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.var(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }
