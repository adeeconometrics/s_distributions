try:
    import numpy as _np
    from math import sqrt as _sqrt, log as _log, exp as _exp
    from typing import Union, Tuple, Dict, List
    from univariate._base import Infinite
except Exception as e:
    print(f"some modules are missing {e}")


class Laplace(Infinite):
    """
    This class contains methods concerning Laplace Distirbution [#]_.

    .. math::
        \\text{Laplace}(x;\\mu, \\b) = \\frac{1}{2b} \\exp{- \\frac{|x - \\mu}{b}}

    Args:

        location(float): location parameter (:math:`\\mu`)
        scale(float): scale parameter (:math:`b`) where scale > 0
        x(float): random variable

    Reference:
        .. [#] Wikipedia contributors. (2020, December 21). Laplace distribution. https://en.wikipedia.org/w/index.php?title=Laplace_distribution&oldid=995563221
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
            either probability density evaluation for some point or plot of Laplace distribution.
        """
        mu = self.location
        b = self.scale
        randvar = self.randvar

        if x is not None:
            if not isinstance(x, (_np.ndarray, List)):
                raise TypeError(
                    f'parameter x only accepts List types or numpy.ndarray')
            else:
                x = _np.array(x)
                (1 / (2 * b)) * _np.exp(_np.abs(x - mu) / b)
        return (1 / (2 * b)) * _exp(abs(randvar - mu) / b)

    def cdf(self, x: Union[List[float], _np.ndarray] = None) -> Union[float, _np.ndarray]:
        """
        Args:

            x (List[float], numpy.ndarray): random variable or list of random variables

        Returns:
            either cumulative distribution evaluation for some point or plot of Laplace distribution.
        """
        mu = self.location
        b = self.scale
        randvar = self.randvar

        def __generator(mu: float, b: float, x: Union[float, _np.ndarray]) -> Union[float, _np.ndarray]:
            return 1 / 2 + ((1 / 2) * _np.sign(x - mu) * (1 - _np.exp(_np.abs(x - mu) / b)))

        if x is not None:
            if not isinstance(x, (_np.ndarray, List)):
                raise TypeError(
                    f'parameter x only accepts List types or numpy.ndarray')
            else:
                x = _np.array(x)
                return __generator(mu, b, x)

        return __generator(mu, b, randvar)

    def mean(self) -> float:
        """
        Returns: Mean of the Laplace distribution.
        """
        return self.location

    def median(self) -> float:
        """
        Returns: Median of the Laplace distribution.
        """
        return self.location

    def mode(self) -> float:
        """
        Returns: Mode of the Laplace distribution.
        """
        return self.location

    def var(self) -> Union[int, float]:
        """
        Returns: Variance of the Laplace distribution.
        """
        return 2 * pow(self.scale, 2)

    def std(self) -> float:
        """
        Returns: Standard deviation of the Laplace distribution
        """
        return _sqrt(self.var())

    def skewness(self) -> float:
        """
        Returns: Skewness of the Laplace distribution.
        """
        return 0.0

    def kurtosis(self) -> float:
        """
        Returns: Kurtosis of the Laplace distribution.
        """
        return 3.0

    def entropy(self) -> float:
        """
        Returns: differential entropy of the Laplace distribution.

        Reference: Park, S.Y. & Bera, A.K.(2009). Maximum entropy autoregressive conditional heteroskedasticity model. Elsivier.
        link: http://wise.xmu.edu.cn/uploadfiles/paper-masterdownload/2009519932327055475115776.pdf
        """
        return 1 + _log(2*self.scale)

    def summary(self) -> Dict[str, Union[float, str]]:
        """
        Returns:
            Dictionary of Laplace distirbution moments. This includes standard deviation. 
        """
        return {
            'mean': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.var(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }
