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
    This class contains methods concerning the Gaussian Distribution.

    Args:

        mean(float): mean of the distribution
        std(float | x>0): standard deviation of the distribution
        randvar(float): random variable

    References:
    - Wikipedia contributors. (2020, December 19). Gaussian distribution. In Wikipedia, The Free Encyclopedia. Retrieved 10:44,
    December 22, 2020, from https://en.wikipedia.org/w/index.php?title=Gaussian_distribution&oldid=995237372
    - Weisstein, Eric W. "Gaussian Distribution." From MathWorld--A Wolfram Web Resource. https://mathworld.wolfram.com/GaussianDistribution.html

    """

    def __init__(self, x: float, mean: float = 0, std_val: float = 1):
        if std_val < 0:
            raise ValueError(
                f"std_val parameter must not be less than 0. Entered value std_val {std_val}")

        self.mean_val = mean
        self.std_val = std_val
        self.randvar = x

    def pdf(self,x: Union[List[float], _np.ndarray] = None) -> Union[float, _np.ndarray]:
        """
        Args:

            x (List[float], numpy.ndarray): random variable or list of random variables

        Returns:
            either probability density evaluation for some point or plot of Gaussian distribution.
        """
        mean = self.mean_val
        std = self.std_val
        randvar = self.randvar

        def __generator(mean, std, x):
            return pow(1 / (std * _sqrt(2 * _pi)), _exp(((x - mean) / 2 * std)**2))

        if x is not None:
            if not isinstance(x, (_np.ndarray, List)):
                raise TypeError(f'parameter x only accepts List types or numpy.ndarray')
            else:
                x = _np.array(x)
                return _np.power(1 / (std * _sqrt(2 * _pi)), _np.exp(((x - mean) / 2 * std)**2))

        return pow(1 / (std * _sqrt(2 * _pi)), _exp(((randvar - mean) / 2 * std)**2))

    def cdf(self, x: Union[List[float], _np.ndarray] = None) -> Union[float, _np.ndarray]:
        """
        Args:

            x (List[float], numpy.ndarray): random variable or list of random variables

        Returns:
            either cumulative distribution evaluation for some point or plot of Gaussian distribution.
        """
        def __generator(mu:float, sig:float, x: Union[float, _np.ndarray]) -> Union[float, _np.ndarray]: 
            return 1/2*(1+_erf((x-mu)/(sig*_sqrt(2))))
            
        if x is not None:
            if not isinstance(x, (_np.ndarray, List)):
                raise TypeError(f'parameter x only accepts List types or numpy.ndarray')
            else:
                x = _np.array(x)
                return __generator(self.mean_val, self.std_val, x)
        return __generator(self.mean_val, self.std_val, self.randvar)

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
        return pow(self.std_val, 2)

    def std(self) -> float:
        """
        Returns: Standard deviation of the Gaussian distribution
        """
        return self.std_val

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
        Summary statistic regarding the Gaussian distribution which contains the following parts of the distribution:
        (mean, median, mode, var, std, skewness, kurtosis).

        Returns:
            Dict[str, Union[float, int]]: [description]
        """
        return {
            'mean': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.var(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }

