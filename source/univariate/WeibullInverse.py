try:
    from scipy.special import gamma as _gamma
    import numpy as _np
    from math import sqrt as _sqrt, log as _log, exp as _exp
    from typing import Union, Dict, List
    from univariate._base import SemiInfinite
except Exception as e:
    print(f"some modules are missing {e}")


class WeilbullInverse(SemiInfinite):
    """
    This class contains methods concerning inverse Weilbull or the Fréchet Distirbution [#]_.

    .. math::
        \\text{WeibullInverse}(x;a,s,m) = \\frac{a}{s} \\Big(\\frac{x-m}{s} \\Big) ^{-1-a} \\exp{\\Big(-\\frac{x-m}{s} \\Big)^{-a}}

    Args:

        shape(float): shape parameter (:math:`a`) where shape >= 0
        scale(float): scale parameter (:math:`s`) where scale >= 0
        location(float): location parameter (:math:`m`)
        randvar(float): random variable where x > location

    Reference:
        .. [#] Wikipedia contributors. (2020, December 7). Fréchet distribution. https://en.wikipedia.org/w/index.php?title=Fr%C3%A9chet_distribution&oldid=992938143
    """

    def __init__(self,  shape: float, scale: float, location: float, randvar: float):
        if shape < 0 or scale < 0:
            raise ValueError(
                f'the value of scale and shape should be greater than 0. Entered values scale was:{scale}, shape:{shape}')
        if randvar < location:
            raise ValueError(
                f'random variable should be greater than the location parameter. Entered values: randvar: {randvar}, location:{location}')
        self.shape = shape
        self.scale = scale
        self.location = location
        self.randvar = randvar

    def pdf(self, x: Union[List[float], _np.ndarray] = None) -> Union[float, _np.ndarray]:
        """
        Args:

            x (List[float], numpy.ndarray): random variable or list of random variables

        Returns:
            either probability density evaluation for some point or plot of Weibull Inverse distribution.
        """
        a = self.shape
        s = self.scale
        m = self.location
        randvar = self.randvar

        if x is not None:
            if not isinstance(x, (_np.ndarray, List)):
                raise TypeError(f'parameter x only accepts List types or numpy.ndarray')
            else:
                x = _np.array(x)
                return (a/s) * _np.power((x-m)/s, -1-a)*_np.exp(-np.power((x-m)/s, -a))
                
        return (a/s) * pow((randvar-m)/s, -1-a)*_exp(-pow((randvar-m)/s, -a))

    def cdf(self, x: Union[List[float], _np.ndarray] = None) -> Union[float, _np.ndarray]:
        """
        Args:

            x (List[float], numpy.ndarray): random variable or list of random variables

        Returns:
            either cumulative distribution evaluation for some point or plot of Weibull Inverse distribution.
        """
        a = self.shape
        s = self.scale
        m = self.location
        randvar = self.randvar
        
        if x is not None:
            if not isinstance(x, (_np.ndarray, List)):
                raise TypeError(f'parameter x only accepts List types or numpy.ndarray')
            else:
                x = _np.array(x)
                return _np.exp(-_np.power((x-m)/s, -a))
                
        return _exp(-pow((x-m)/s, -a))

    def mean(self) -> float:
        """
        Returns: Mean of the Fréchet distribution.
        """
        if self.shape > 1:
            return self.location + (self.scale*_gamma(1 - 1/self.shape))
        return _np.inf

    def median(self) -> float:
        """
        Returns: Median of the Fréchet distribution.
        """
        return self.location + (self.scale/pow(_log(2), 1/self.shape))

    def mode(self) -> float:
        """
        Returns: Mode of the Fréchet distribution.
        """
        return self.location + self.scale*(self.shape/pow(1 + self.shape, 1/self.shape))

    def var(self) -> Union[float, str]:
        """
        Returns: Variance of the Fréchet distribution.
        """
        a = self.shape
        s = self.scale
        if a > 2:
            return pow(s, 2)*(_gamma(1-2/a)-pow(_gamma(1-1/a), 2))
        return "infinity"

    def std(self) -> Union[float, str]:
        """
        Returns: Standard devtiation of the Fréchet distribution.
        """
        if self.var() == "infinity":
            return "infinity"
        return _sqrt(self.var())

    def skewness(self) -> Union[float, str]:
        """
        Returns: Skewness of the Fréchet distribution.
        """
        a = self.shape
        if a > 3:
            return (_gamma(1-3/a)-3*_gamma(1-2/a)*_gamma(1-1/a)+2*_gamma(1-1/a)**3)/pow(_gamma(1-2/a)-pow(_gamma(1-1/a), 2), 3/2)
        return "infinity"

    def kurtosis(self) -> Union[float, str]:
        """
        Returns: Kurtosis of the Fréchet distribution.
        """
        a = self.shape
        if a > 4:
            return -6+(_gamma(1-4/a)-4*_gamma(1-3/a)*_gamma(1-1/a)+3*pow(_gamma(1-2/a), 2))/pow(_gamma(1-2/a)-pow(_gamma(1-1/a), 2), 2)
        return "infinity"

    def summary(self) -> Dict[str, Union[float, int, str]]:
        """
        Returns:
            Dictionary of Fréchet distirbution moments. This includes standard deviation. 
        """
        return {
            'mean': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.var(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }

