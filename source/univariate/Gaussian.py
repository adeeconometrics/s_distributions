# Test Gaussian PDF

try:
    from scipy.special import erf as _erf
    import numpy as _np
    from math import sqrt as _sqrt, log as _log, pi as _pi, e as _e, exp as _exp
    from typing import Union, Dict, List
    from univariate._base import Infinite
except Exception as e:
    print(f"some modules are missing {e}")


class Gaussian(Infinite):
    """
    This class contains methods concerning the Gaussian Distribution [#]_ [#]_.

    .. math::
        \\text{Gaussian}(x;\\mu,\\sigma) = \\frac{1}{\\sigma \\sqrt(2 \\pi)} e^{\\frac{1}{2}\\big( \\frac{x-\\mu}{\\sigma}\\big)^2}

    Args:

        mean(float): mean of the distribution (:math:`\\mu`)
        std(float): standard deviation (:math:`\\sigma`) of the distribution where std > 0
        x(float): random variable 

    References:
        .. [#] Wikipedia contributors. (2020, December 19). Gaussian distribution. https://en.wikipedia.org/w/index.php?title=Gaussian_distribution&oldid=995237372
        .. [#] Weisstein, Eric W. "Gaussian Distribution." From MathWorld--A Wolfram Web Resource. https://mathworld.wolfram.com/GaussianDistribution.html

    """

    def __init__(self, mean: float = 0, stdev: float = 1):
        if stdev < 0:
            raise ValueError("stdev parameter must not be less than 0.")

        self.mean_val = mean
        self.stdev = stdev

    def pdf(self, x: Union[List[float], _np.ndarray, float]) -> Union[float, _np.ndarray]:
        """
        Args:
            x (Union[List[float], numpy.ndarray, float]): random variable(s)

        Returns:
            Union[float, numpy.ndarray]: evaluation of pdf at x
        """
        mean = self.mean_val
        std = self.stdev

        if isinstance(x, (_np.ndarray, List)):
            x = _np.array(x)
            return _np.power(1 / (std * _sqrt(2 * _pi)), _np.exp(((x - mean) / 2 * std)**2))

        return pow(1 / (std * _sqrt(2 * _pi)), _exp(((x - mean) / 2 * std)**2))

    def cdf(self, x: Union[List[float], _np.ndarray, float]) -> Union[float, _np.ndarray]:
        """
        Args:
            x (Union[List[float], numpy.ndarray, float]): data point(s) of interest

        Returns:
            Union[float, numpy.ndarray]: evaluation of cdf at x
        """
        def __generator(mu: float, sig: float, x: Union[float, _np.ndarray]) -> Union[float, _np.ndarray]:
            return 1/2*(1+_erf((x-mu)/(sig*_sqrt(2))))

        if isinstance(x, (_np.ndarray, List)):
            x = _np.array(x)
            return __generator(self.mean_val, self.stdev, x)
        return __generator(self.mean_val, self.stdev, x)

    def mean(self) -> float:
        """
        Returns: Mean of the Gaussian distribution
        """
        return self.mean_val

    def median(self) -> float:
        """
        Returns: Median of the Gaussian distribution
        """
        return self.mean_val

    def mode(self) -> float:
        """
        Returns: Mode of the Gaussian distribution
        """
        return self.mean_val

    def var(self) -> float:
        """
        Returns: Variance of the Gaussian distribution
        """
        return pow(self.stdev, 2)

    def std(self) -> float:
        """
        Returns: Standard deviation of the Gaussian distribution
        """
        return self.stdev

    def skewness(self) -> float:
        """
        Returns: Skewness of the Gaussian distribution
        """
        return 0.0

    def kurtosis(self) -> float:
        """
        Returns: Kurtosis of the Gaussian distribution
        """
        return 0.0

    def entropy(self) -> float:
        """
        Returns: differential entropy of the Gaussian distribution

        Reference: Park, S.Y. & Bera, A.K.(2009). Maximum entropy autoregressive conditional heteroskedasticity model. Elsivier.
        link: http://wise.xmu.edu.cn/uploadfiles/paper-masterdownload/2009519932327055475115776.pdf
        """
        return _log(self.std()*_sqrt(2 * _pi * _e))

    def summary(self) -> Dict[str, Union[float, int, str]]:
        """
        Returns:
            Dictionary of Gaussian distirbution moments. This includes standard deviation. 
        """
        return {
            'mean': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.var(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }
