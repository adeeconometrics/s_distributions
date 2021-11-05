try:
    from scipy.special import gamma as _gamma
    from numpy import euler_gamma as _euler_gamma
    import numpy as _np
    from math import sqrt as _sqrt, log as _log
    from typing import Union, Tuple, Dict, List
    from univariate._base import SemiInfinite
except Exception as e:
    print(f"some modules are missing {e}")


class Weibull(SemiInfinite):
    """
    This class contains methods concerning Weibull Distirbution. Also known as Fréchet distribution.
    Args:

        shape(float | [0, infty)): mean parameter
        scale(float | [0, infty)): standard deviation
        randvar(float | [0, infty)): random variable. Optional. Use when cdf and pdf or p value of interest is desired.

    Reference:
    - Wikipedia contributors. (2020, December 13). Weibull distribution. In Wikipedia, The Free Encyclopedia.
    Retrieved 11:32, December 28, 2020, from https://en.wikipedia.org/w/index.php?title=Weibull_distribution&oldid=993879185
    """

    def __init__(self, shape: float, scale: float, randvar: float = 0.5):
        if shape < 0 or scale < 0 or randvar < 0:
            raise ValueError(
                f'all parameters should be a positive number. Entered values: shape: {shape}, scale{scale}, randvar{randvar}')
        self.scale = scale
        self.shape = shape
        self.randvar = randvar

    def pdf(self, x: Union[List[float], _np.ndarray] = None) -> Union[float, _np.ndarray, List[float]]:
        """
        Args:

            x (List[float], numpy.ndarray): random variable or list of random variables

        Returns:
            either probability density evaluation for some point or plot of Weibull distribution.
        """
        scale = self.scale
        shape = self.shape
        randvar = self.randvar

        def __generator(_lambda:float, k:float, x:float) -> float:
            if x < 0:
                return 0.0
            if x >= 0:
                return pow((k/_lambda)*(x/_lambda), k-1)*_np.exp(-pow(x/_lambda, k))

        if x is not None:
            if not isinstance(x, (_np.ndarray, List)):
                raise TypeError(f'parameter x only accepts List types or numpy.ndarray')
            else:
                return [__generator(scale, shape, i) for i in x]

        return __generator(scale, shape, randvar)

    def cdf(self, x: Union[List[float], _np.ndarray] = None) -> Union[float, _np.ndarray, List[float]]:
        """
        Args:

            x (List[float], numpy.ndarray): random variable or list of random variables

        Returns:
            either cumulative distribution evaluation for some point or plot of Weibull distribution.
        """
        scale = self.scale
        shape = self.shape
        randvar = self.randvar

        def __generator(_lambda:float, k:float, x:float) -> float:
            if x < 0:
                return 0.0
            if x >= 0:
                return 1-_np.exp(-pow(x/_lambda, k))

        if x is not None:
            if not isinstance(x, (_np.ndarray, List)):
                raise TypeError(f'parameter x only accepts List types or numpy.ndarray')
            else:
                return [__generator(scale, shape, i) for i in x]

        return __generator(scale, shape, randvar)

    def mean(self) -> float:
        """
        Returns: Mean of the Weibull distribution.
        """
        return self.scale*_gamma(1+(1/self.shape))

    def median(self) -> float:
        """
        Returns: Median of the Weibull distribution.
        """
        return self.scale*pow(_log(2), 1/self.shape)

    def mode(self) -> float:
        """
        Returns: Mode of the Weibull distribution.
        """
        if self.shape > 1:
            return self.scale*pow((self.shape-1)/self.shape, 1/self.shape)
        return 0

    def var(self) -> float:
        """
        Returns: Variance of the Weibull distribution.
        """
        return pow(self.scale, 2) * pow(_gamma(1+2/self.shape) - _gamma(1+1/self.shape), 2)

    def std(self) -> float:
        """
        Returns: Standard deviation of the Weilbull distribution
        """
        return _sqrt(pow(self.scale, 2) * pow(_gamma(1+2/self.shape) - _gamma(1+1/self.shape), 2))

    def entropy(self) -> float:
        """
        Returns: differential entropy of the Weilbull distribution.

        Reference: Park, S.Y. & Bera, A.K.(2009). Maximum entropy autoregressive conditional heteroskedasticity model. Elsivier.
        link: http://wise.xmu.edu.cn/uploadfiles/paper-masterdownload/2009519932327055475115776.pdf
        """
        return (self.scale+1) * _euler_gamma/self.scale + _log(self.shape/self.scale) + 1

    def summary(self) -> Dict[str, Union[float, int, str]]:
        """
        Summary statistic regarding the ChiSquare-distribution which contains the following parts of the distribution:
        (mean, median, mode, var, std, skewness, kurtosis).

        Returns:
            Dict[str, Union[float, int, str]]
        """
        return {
            'mean': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.var(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }

