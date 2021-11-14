try:
    from scipy.special import gamma as _gamma
    from numpy import euler_gamma as _euler_gamma
    import numpy as _np
    from math import sqrt as _sqrt, log as _log, exp as _exp
    from typing import Union, Dict, List
    from univariate._base import SemiInfinite
except Exception as e:
    print(f"some modules are missing {e}")


class Weibull(SemiInfinite):
    """
    This class contains methods concerning Weibull Distirbution [#]_.

    .. math::
        \\text{Weibull}(x;\\lambda, k)  = \\frac{k}{\\lambda} \\Big( \\frac{x}{\\lambda}\\Big)^{k-1} \\exp(-(x/\\lambda)^k)

    Args:

        shape(float): shape parameter (:math:`\\lambda`) where shape >= 0
        scale(float): scale parameter (:math:`k`) where scale >= 0
        randvar(float): random variable where x >= 0

    Reference:
        .. [#] Wikipedia contributors. (2020, December 13). Weibull distribution. https://en.wikipedia.org/w/index.php?title=Weibull_distribution&oldid=993879185
    """

    def __init__(self, shape: float, scale: float):
        if shape < 0 or scale < 0:
            raise ValueError('all parameters should be a positive number.')
        self.scale = scale
        self.shape = shape

    def pdf(self, x: Union[List[float], _np.ndarray, float]) -> Union[float, _np.ndarray]:
        """
        Args:

            x (List[float], numpy.ndarray): random variable or list of random variables

        Returns:
            either probability density evaluation for some point or plot of Weibull distribution.
        """
        scale = self.scale
        shape = self.shape

        if isinstance(x, (_np.ndarray, List)):
            if not type(x) is _np.ndarray:
                x = _np.array(x)

            def f1(x): return 0.0
            def f2(x): return _np.power(shape/scale*x/scale, shape-1) * \
                _np.exp(-_np.power(x/scale, shape))
            return _np.piecewise(x, [x < 0, x >= 0], [f1, f2])

        return pow((shape/scale)*(x/scale), shape-1)*_exp(-pow(x/scale, shape)) if x >= 0 else 0.0

    def cdf(self, x: Union[List[float], _np.ndarray, float]) -> Union[float, _np.ndarray]:
        """
        Args:

            x (List[float], numpy.ndarray): random variable or list of random variables

        Returns:
            either cumulative distribution evaluation for some point or plot of Weibull distribution.
        """
        scale = self.scale
        shape = self.shape

        if isinstance(x, (_np.ndarray, List)):
            if not type(x) is _np.ndarray:
                x = _np.array(x)

            def f1(x): return 1 - _np.exp(-_np.power(x/scale, shape))
            def f2(x): return 0.0
            return _np.piecewise(x, [x >= 0, x < 0], [f1, f2])

        return 1-_exp(-pow(x/scale, shape)) if x >= 0 else 0.0

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
        Returns:
            Dictionary of Weibull distirbution moments. This includes standard deviation. 
        """
        return {
            'mean': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.var(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }
