try:
    import numpy as _np
    from math import sqrt as _sqrt, pi as _pi, exp as _exp
    from typing import Union, Dict, List
    from univariate._base import Infinite
except Exception as e:
    print(f"some modules are missing {e}")


class Logistic(Infinite):
    """
    This class contains methods concerning Logistic Distirbution [#]_.

    .. math::
        \\text{Logistic}(x;\\mu,s) = \\frac{\\exp{(-(x-\\mu)/s)}} {s(1+\\exp(-(x-\\mu)/s)^2)}

    Args:

        location(float): location parameter (:math:`\\mu`)
        scale(float): scale parameter (:math:`s`) x > 0 
        x(float): random variable

    Reference:
        .. [#] Wikipedia contributors. (2020, December 12). Logistic distribution. https://en.wikipedia.org/w/index.php?title=Logistic_distribution&oldid=993793195
    """

    def __init__(self, location: float, scale: float, randvar: float):
        if scale < 0:
            raise ValueError(
                f'scale should be greater than 0. Entered value for Scale:{scale}')

        self.scale = scale
        self.location = location
        self.randvar = randvar

    def pdf(self, x: Union[List[float], _np.ndarray] = None) -> Union[float, _np.ndarray]:
        """
        Args:

            x (List[float], numpy.ndarray): random variable or list of random variables

        Returns:
            either probability density evaluation for some point or plot of Logistic distribution.
        """
        mu = self.location
        s = self.scale
        randvar = self.randvar

        def __generator(mu, s, x): return _np.exp(-(x - mu) / s) / (s * (1 + _np.exp(
            -(x - mu) / s))**2)

        if x is not None:
            if not isinstance(x, (_np.ndarray, List)):
                raise TypeError(
                    f'parameter x only accepts List types or numpy.ndarray')
            else:
                x = _np.array(x)
                return _np.exp(-(x - mu) / s) / (s * (1 + _np.power(_np.exp(-(x - mu) / s)), 2))
        return _exp(-(randvar - mu) / s) / (s * (1 + pow(_exp(-(randvar - mu) / s)), 2))

    def cdf(self, x: Union[List[float], _np.ndarray] = None) -> Union[float, _np.ndarray]:
        """
        Args:

            x (List[float], numpy.ndarray): random variable or list of random variables

        Returns:
            either cumulative distribution evaluation for some point or plot of Logistic distribution.
        """
        mu = self.location
        s = self.scale
        randvar = self.randvar

        if x is not None:
            if not isinstance(x, (_np.ndarray, List)):
                raise TypeError(
                    f'parameter x only accepts List types or numpy.ndarray')
            else:
                x = _np.array(x)
                return 1 / (1 + _np.exp(-(x - mu) / s))
        return 1 / (1 + _exp(-(randvar - mu) / s))

    def mean(self) -> float:
        """
        Returns: Mean of the Logistic distribution.
        """
        return self.location

    def median(self) -> float:
        """
        Returns: Median of the Logistic distribution.
        """
        return self.location

    def mode(self) -> float:
        """
        Returns: Mode of the Logistic distribution.
        """
        return self.location

    def var(self) -> float:
        """
        Returns: Variance of the Logistic distribution.
        """
        return pow(self.scale, 2) * pow(_pi, 2)/3

    def std(self) -> float:
        """
        Returns: Standard deviation of the Logistic distribution.
        """
        return _sqrt(self.var())

    def skewness(self) -> float:
        """
        Returns: Skewness of the Logistic distribution.
        """
        return 0.0

    def kurtosis(self) -> float:
        """
        Returns: Kurtosis of the Logistic distribution.
        """
        return 6 / 5

    def entropy(self) -> float:
        """
        Returns: differential entropy of the Logistic distribution.

        Reference: Park, S.Y. & Bera, A.K.(2009). Maximum entropy autoregressive conditional heteroskedasticity model. Elsivier.
        link: http://wise.xmu.edu.cn/uploadfiles/paper-masterdownload/2009519932327055475115776.pdf
        """
        return 2.0

    def summary(self) -> Dict[str, Union[float, str]]:
        """
        Returns:
            Dictionary of Logistic distirbution moments. This includes standard deviation. 
        """
        return {
            'mean': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.var(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }
