try:
    from scipy.special import erfc as _erfc
    import numpy as _np
    from math import sqrt as _sqrt, log as _log, pi as _pi, exp as _exp
    from typing import Union, Tuple, Dict, List
    from univariate._base import SemiInfinite
except Exception as e:
    print(f"some modules are missing {e}")


class LogNormal(SemiInfinite):
    """
    This class contains methods concerning the Log Normal Distribution [#]_ [#]_.

    .. math::
        \\text{LogNormal}(x;\\mu,\\sigma) = \\frac{1}{x\\sigma\\sqrt{2\\pi}} \\exp{\\Big( - \\frac{(\\ln x - \\mu)^2}{2\\sigma^2} \\Big)}

    Args:

        mean (float): mean parameter (:math:`\\mu`)
        std (float): standard deviation (:math:`\\sigma`) where std > 0
        x (float): random variable where x >= 0

    References:
        .. [#] Weisstein, Eric W. "Log Normal Distribution." From MathWorld--A Wolfram Web Resource.https://mathworld.wolfram.com/LogNormalDistribution.html
        .. [#] Wikipedia contributors. (2020, December 18). Log-normal distribution. https://en.wikipedia.org/w/index.php?title=Log-normal_distribution&oldid=994919804
    """

    def __init__(self, mean: float, std: float, randvar: float):
        if randvar < 0:
            raise ValueError('random variable should be greater than 0.')
        if std < 0:
            raise ValueError('random variable should be greater than 0.')

        self.randvar = randvar
        self.mean_val = mean
        self.stdev = std

    def pdf(self, x: Union[List[float], _np.ndarray, float]) -> Union[float, _np.ndarray]:
        """
        Args:
            x (Union[List[float], numpy.ndarray, float]): random variable(s)

        Raises:
            ValueError: when there exist a value of x < 0

        Returns:
            Union[float, numpy.ndarray]: evaluation of pdf at x
        """
        mean = self.mean
        stdev = self.stdev

        if isinstance(x, (_np.ndarray, List)):
            x = _np.array(x)
            if _np.any(x < 0):
                raise ValueError('random variable should be greater than 0.')
            return 1 / (x * stdev * _sqrt(2 * _pi)) * _np.exp(-(_np.log(x - mean)**2) / (2 * stdev**2))

        if x < 0:
            raise ValueError('random variable should be greater than 0.')
        return 1 / (x * stdev * _sqrt(2 * _pi)) * _exp(-(_log(x - mean)**2) / (2 * stdev**2))

    def cdf(self, x: Union[List[float], _np.ndarray, float]) -> Union[float, _np.ndarray]:
        """
        Args:
            x (Union[List[float], numpy.ndarray, float]): data point(s) of interest

        Returns:
            Union[float, numpy.ndarray]: evaluation of cdf at x
        """
        mean = self.mean
        std = self.std

        if isinstance(x, (_np.ndarray, List)):
            x = _np.array(x)
            return 0.5 + 0.5*_erfc(-_np.log(x - mean)/(std * _sqrt(2)))

        return 0.5 + 0.5*_erfc(-_np.log(x - mean)/(std * _sqrt(2)))

    def mean(self) -> float:
        """
        Returns: Mean of the log normal distribution.
        """
        return _exp(self.mean_val + pow(self.stdev, 2) / 2)

    def median(self) -> float:
        """
        Returns: Median of the log normal distribution.
        """
        return _exp(self.mean_val)

    def mode(self) -> float:
        """
        Returns: Mode of the log normal distribution.
        """
        return _exp(self.mean_val - pow(self.stdev, 2))

    def var(self) -> float:
        """
        Returns: Variance of the log normal distribution.
        """
        std = self.stdev
        mean = self.mean_val
        return (_exp(pow(std, 2)) - 1) * _exp(2 * mean + pow(std, 2))

    def std(self) -> float:
        """
        Returns: Standard deviation of the log normal distribution
        """
        return self.stdev

    def skewness(self) -> float:
        """
        Returns: Skewness of the log normal distribution.
        """
        std = self.stdev
        mean = self.mean_val
        return (_exp(pow(std, 2)) + 2) * _sqrt(_exp(pow(std, 2)) - 1)

    def kurtosis(self) -> float:
        """
        Returns: Kurtosis of the log normal distribution.
        """
        std = self.stdev
        return _exp(
            4 * pow(std, 2)) + 2 * _exp(3 * pow(std, 2)) + 3 * _exp(2 * pow(std, 2)) - 6

    def entropy(self) -> float:
        """
        Returns: differential entropy of the log normal distribution.

        Reference: Park, S.Y. & Bera, A.K.(2009). Maximum entropy autoregressive conditional heteroskedasticity model. Elsivier.
        link: http://wise.xmu.edu.cn/uploadfiles/paper-masterdownload/2009519932327055475115776.pdf
        """
        return self.mean_val + 0.5 * _log(2*_pi*_e*self.stdev**2)

    def summary(self) -> Dict[str, Union[float, int, str]]:
        """
        Returns:
            Dictionary of Log Normal distirbution moments. This includes standard deviation. 
        """
        return {
            'mean': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.var(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }
