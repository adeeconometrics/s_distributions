try:
    from scipy.special import logit as _logit, erf as _erf
    import numpy as _np
    from math import sqrt as _sqrt, pi as _pi, exp as _exp
    from typing import Union, Tuple, Dict, List
    from univariate._base import BoundedInterval
except Exception as e:
    print(f"some modules are missing {e}")


class LogitNormal(BoundedInterval):
    """
    This class contains methods concerning Logit Normal Distirbution [#]_.

    .. math::
        \\text{LogitNormal}(x;\\mu,\\sigma) = \\frac{1}{\\sigma \\sqrt(2\\pi) \\cdot x(1-x)} \\exp{\\Big(-\\frac{(logit(x)-\\mu)^2}{2\\sigma^2} \\Big)}

    Args:

        sq_scale (float): squared scale parameter
        location(float): location parameter
        x(float): random variable where x is between 0 and 1

    Reference:
        .. [#] Wikipedia contributors. (2020, December 9). Logit-normal distribution. https://en.wikipedia.org/w/index.php?title=Logit-normal_distribution&oldid=993237113
    """

    def __init__(self, sq_scale: float, location: float):
        self.sq_scale = sq_scale
        self.location = location

    def pdf(self, x: Union[List[float], _np.ndarray, float]) -> Union[float, _np.ndarray]:
        """
        Args:
            x (Union[List[float], numpy.ndarray, float]): random variable(s)

        Raises:
            ValueError: when there exist a value below 0 and greater than 1

        Returns:
            Union[float, numpy.ndarray]: evaluation of pdf at x
        """
        mu = self.location
        sig = self.sq_scale

        if isinstance(x, (_np.ndarray, List)):
            x = _np.array(x)
            if _np.any(_np.logical_or(x < 0, x > 1)):
                raise ValueError(
                    'random variable should only be in between 0 and 1')
            return (1/(sig*_sqrt(2*_pi))) * _np.exp(-(_np.power(_logit(x)-mu, 2)/(2*pow(sig, 2)))) * 1/(x*(1-x))

        if x < 0 or x > 1:
            raise ValueError(
                'random variable should only be in between 0 and 1')
        return (1/(sig*_sqrt(2*_pi))) * _exp(-pow(_logit(x)-mu, 2)/(2*pow(sig, 2))) * 1/(x*(1-x))

    def cdf(self, x: Union[List[float], _np.ndarray, float]) -> Union[float, _np.ndarray]:
        """
        Args:
            x (Union[List[float], numpy.ndarray, float]): data point(s) of interest

        Returns:
            Union[float, numpy.ndarray]: evaluation of cdf at x
        """
        mu = self.location
        sig = self.sq_scale

        def __generator(mu: float, sig: float, x: Union[float, _np.ndarray]) -> Union[float, _np.ndarray]:
            return 0.5 * (1+_erf((_logit(x)-mu)/_sqrt(2*pow(sig, 2))))

        if isinstance(x, (_np.ndarray, List)):
            x = _np.array(x)
            return __generator(mu, sig, x)

        return __generator(mu, sig, x)

    def mean(self) -> str:
        """
        Returns: Mean of the Logit Normal distribution.
        """
        return "no analytical solution"

    def mode(self) -> str:
        """
        Returns: Mode of the Logit Normal distribution.
        """
        return "no analytical solution"

    def var(self) -> str:
        """
        Returns: Variance of the Logit Normal distribution.
        """
        return "no analytical solution"

    def std(self) -> str:
        """
        Returns: Standard deviation of the Logit Normal distribution.
        """
        return "no analytical solution"

    def entropy(self) -> str:
        """
        Returns: differential entropy of Logit Normal distribution.

        Reference: Park, S.Y. & Bera, A.K.(2009). Maximum entropy autoregressive conditional heteroskedasticity model. Elsivier.
        link: http://wise.xmu.edu.cn/uploadfiles/paper-masterdownload/2009519932327055475115776.pdf
        """
        return "unsupported"

    def summary(self) -> Dict[str, str]:
        """
        Returns:
            Dictionary of Logit Normal distirbution moments. This includes standard deviation. 
        """
        return {
            'mean': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.var(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }
