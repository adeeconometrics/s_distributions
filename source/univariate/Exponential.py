try:
    import numpy as _np
    from typing import Union, Dict, List
    from math import sqrt as _sqrt, log as _log, exp as _exp
    from univariate._base import SemiInfinite
except Exception as e:
    print(f"some modules are missing {e}")


class Exponential(SemiInfinite):
    """
    This class contans methods for evaluating Exponential Distirbution [#]_ [#]_.

    .. math:: \\text{Exponential}(x;\\lambda) = \\lambda e^{-\\lambda x}

    Args:

        - rate (float): rate parameter (:math:`\\lambda`) where rate > 0
        - x (float): random variable where x > 0

    References:
        .. [#] Weisstein, Eric W. "Exponential Distribution." From MathWorld--A Wolfram Web Resource. https://mathworld.wolfram.com/ExponentialDistribution.html
        .. [#] Wikipedia contributors. (2020, December 17). Exponential distribution. https://en.wikipedia.org/w/index.php?title=Exponential_distribution&oldid=994779060
    """

    def __init__(self, rate: float):
        if rate < 0:
            raise ValueError(f'lambda parameter should be greater than 0.')

        self.rate = rate

    def pdf(self, x: Union[List[float], _np.ndarray, float]) -> Union[float, _np.ndarray]:
        """
        Args:
            x (Union[List[float], numpy.ndarray, float]): random variable(s)

        Returns:
            Union[float, numpy.ndarray]: evaluation of pdf at x
        """
        rate = self.rate

        if isinstance(x, (_np.ndarray, List)):
            x = _np.array(x)
            return _np.piecewise(x, [x >= 0, x < 0], [lambda x: rate*_np.exp(-(rate*(x))), lambda x: 0.0])

        return rate*_exp(-rate*x) if x >= 0 else 0.0

    def cdf(self, x: Union[List[float], _np.ndarray, float]) -> Union[float, _np.ndarray]:
        """
        Args:
            x (Union[List[float], numpy.ndarray, float]): data point(s) of interest

        Returns:
            Union[float, numpy.ndarray]: evaluation of cdf at x
        """
        rate = self.rate

        if isinstance(x, (_np.ndarray, List)):
            return _np.piecewise(x, [x > 0, x <= 0], [lambda x: 1 - _np.exp(-rate*x), lambda x: 0.0])

        return 1 - _exp(-rate*x) if x > 0 else 0.0

    def mean(self) -> float:
        """
        Returns: Mean of the Exponential distribution
        """
        return 1 / self.rate

    def median(self) -> float:
        """
        Returns: Median of the Exponential distribution
        """
        return _log(2) / self.rate

    def mode(self) -> float:
        """
        Returns: Mode of the Exponential distribution
        """
        return 0.0

    def var(self) -> float:
        """
        Returns: Variance of the Exponential distribution
        """
        return 1 / pow(self.rate, 2)

    def std(self) -> float:
        """
        Returns: Standard deviation of the Exponential distribution
        """
        return _sqrt(self.var())

    def skewness(self) -> float:
        """
        Returns: Skewness of the Exponential distribution
        """
        return 2.0

    def kurtosis(self) -> float:
        """
        Returns: Kurtosis of the Exponential distribution
        """
        return 6.0

    def entorpy(self) -> float:
        """
        Returns: differential entropy of the Exponential distribution

        Reference: Park, S.Y. & Bera, A.K.(2009). Maximum entropy autoregressive conditional heteroskedasticity model. Elsivier.
        link: http://wise.xmu.edu.cn/uploadfiles/paper-masterdownload/2009519932327055475115776.pdf
        """
        return 1 - _log(self.rate)

    def summary(self) -> Dict[str, Union[float, int]]:
        """
        Returns:
            Dictionary of Exponential distirbution moments. This includes standard deviation. 
        """
        return {
            'mean': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.var(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }
