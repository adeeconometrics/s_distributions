try:
    from scipy.special import gamma as _gamma, gammainc as _gammainc, digamma as _digamma
    import numpy as _np
    from math import sqrt as _sqrt, log as _log, exp as _exp
    from typing import Union, Dict, List
    from univariate._base import SemiInfinite
except Exception as e:
    print(f"some modules are missing {e}")

class Gamma(SemiInfinite):
    """
    This class contains methods concerning a variant of Gamma distribution [#]_.

    .. math:: 
        \\text{Gamma}(x;a,b) = \\frac{1}{b^a \\Gamma(a)} \\ x^{a-1} e^{\\frac{-x}{b}}

    Args:

        shape(float): shape parameter (:math:`a`) where shape > 0
        scale(float): scale parameter (:math:`b`) where scale > 0
        x(float): random variable where x > 0

    References:
        .. [#] Matlab(2020). Gamma Distribution. https://www.mathworks.com/help/stats/gamma-distribution.html
    """

    def __init__(self, shape: float, b: float, x: float):
        if shape < 0:
            raise ValueError(
                f'shape should be greater than 0. Entered value for shape:{shape}')
        if b < 0:
            raise ValueError(
                f'scale should be greater than 0. Entered value for b:{b}')
        if x < 0:
            raise ValueError(
                f'random variable should be greater than 0. Entered value for x:{b}')
        self.shape = shape
        self.scale = b
        self.x = x

    def pdf(self, x: Union[List[float], _np.ndarray] = None) -> Union[float, _np.ndarray]:
        """
        Args:

            x (List[float], numpy.ndarray): random variable or list of random variables

        Returns:
            either probability density evaluation for some point or plot of Gamma distribution.
        """
        shape = self.shape
        scale = self.scale
        randvar = self.x

        if x is not None:
            if not isinstance(x, (_np.ndarray,List)):
                raise TypeError(
                    f'parameter x only accepts List types or numpy.ndarray')
            else:
                x = _np.array(x)
                return (1 / (pow(scale, shape) * _gamma(shape))) * _np.log(x, shape - 1) * _np.exp(-x / scale)
        return (1 / (pow(scale, shape) * _gamma(shape))) * _log(randvar, shape - 1) * _exp(-randvar / scale)

    def cdf(self, x: Union[List[float], _np.ndarray] = None) -> Union[float, _np.ndarray]:
        """
        Args:

            x (List[float], numpy.ndarray): random variable or list of random variables

        Returns:
            either cumulative distribution evaluation for some point or plot of Gamma distribution.
        """
        shape = self.shape
        scale = self.scale
        randvar = self.x

        # there is no apparent explanation for reversing gammainc's parameter, but it works quite perfectly in my prototype
        def __generator(shape: float, b: float, x: Union[float, _np.ndarray]) -> Union[float, _np.ndarray]:
            return 1 - _gammainc(shape, x / b)

        if x is not None:
            if not isinstance(x, (_np.ndarray,List)):
                raise TypeError(
                    f'parameter x only accepts List types or numpy.ndarray')
            else:
                x = _np.array(x)
                return __generator(shape, scale, x)
        return __generator(shape, scale, randvar)

    def mean(self) -> Union[float, int]:
        """
        Returns: Mean of the Gamma distribution
        """
        return self.shape * self.scale

    def median(self) -> str:
        """
        Returns: Median of the Gamma distribution.
        """
        return "No simple closed form."

    def mode(self) -> Union[float, int]:
        """
        Returns: Mode of the Gamma distribution
        """
        return (self.shape - 1) * self.scale

    def var(self) -> Union[float, int]:
        """
        Returns: Variance of the Gamma distribution
        """
        return self.shape * pow(self.scale, 2)

    def std(self) -> float:
        """
        Returns: Standard deviation of the Gamma distribution
        """
        return _sqrt(self.var())

    def skewness(self) -> float:
        """
        Returns: Skewness of the Gamma distribution
        """
        return 2 / _sqrt(self.shape)

    def kurtosis(self) -> float:
        """
        Returns: Kurtosis of the Gamma distribution
        """
        return 6 / self.shape

    def entropy(self) -> float:
        """
        Returns: differential entropy of the Gamma distribution

        Reference: Park, S.Y. & Bera, A.K.(2009). Maximum entropy autoregressive conditional heteroskedasticity model. Elsivier.
        link: http://wise.xmu.edu.cn/uploadfiles/paper-masterdownload/2009519932327055475115776.pdf
        """
        k = self.shape
        theta = self.scale
        return k + _log(theta)+_log(_gamma(k))-(1-k)*_digamma(k)

    def summary(self) -> Dict[str, Union[float, int, str]]:
        """
        Returns:
            Dictionary of Gamma distirbution moments. This includes standard deviation. 
        """
        return {
            'mean': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.var(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }
