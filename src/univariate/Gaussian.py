# Test Gaussian PDF

try:
    from scipy.special import erf as _erf
    import numpy as _np
    from math import sqrt as _sqrt, log as _log, pi as _pi, e as _e, exp as _exp
    from typing import Union, Tuple, Dict, List
    from _base import Infinite
except Exception as e:
    print(f"some modules are missing {e}")


class Gaussian(Infinite):
    """
    This class contains methods concerning the Gaussian Distribution.

    Args:

        mean(float): mean of the distribution
        std(float | x>0): standard deviation of the distribution
        randvar(float): random variable

    Methods:

        - pdf for probability density function.
        - cdf for cumulative distribution function.
        - pvalue for p-values.
        - mean for evaluating the mean of the distribution.
        - median for evaluating the median of the distribution.
        - mode for evaluating the mode of the distribution.
        - var for evaluating the variance of the distribution.
        - std for evaluating the standard deviation of the distribution.
        - skewness for evaluating the skewness of the distribution.
        - kurtosis for evaluating the kurtosis of the distribution.
        - entropy for differential entropy of the distribution.
        - summary for printing the summary statistics of the distribution.
        - keys for returning a dictionary of summary statistics.

    References:
    - Wikipedia contributors. (2020, December 19). Gaussian distribution. In Wikipedia, The Free Encyclopedia. Retrieved 10:44,
    December 22, 2020, from https://en.wikipedia.org/w/index.php?title=Gaussian_distribution&oldid=995237372
    - Weisstein, Eric W. "Gaussian Distribution." From MathWorld--A Wolfram Web Resource. https://mathworld.wolfram.com/GaussianDistribution.html

    """

    def __init__(self, x: float, mean: Union[int, float] = 0, std_val: Union[int, float] = 1):
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
            if not (isinstance(x, _np.ndarray)) and issubclass(x, List):
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
            if not (isinstance(x, _np.ndarray)) and issubclass(x, List):
                raise TypeError(f'parameter x only accepts List types or numpy.ndarray')
            else:
                x = _np.array(x)
                return __generator(self.mean_val, self.std_val, x)
        return __generator(self.mean_val, self.std_val, self.randvar)

    def p_val(self, x_lower=-_np.inf, x_upper=None):
        """
        Args:

            x_lower(float): defaults to -_np.inf. Defines the lower value of the distribution. Optional.
            x_upper(float | x_upper>x_lower): defaults to None. If not defined defaults to random variable x. Optional.

            Note: definition of x_lower and x_upper are only relevant when probability is between two random variables.
            Otherwise, the default random variable is x.

        Returns:
            p-value of the Gaussian distribution evaluated at some random variable.
        """
        def __cdf(mu, sig, x): return 1/2*(1+_erf((x-mu)/(sig*_sqrt(2))))
        if x_upper != None:
            if x_lower > x_upper:
                raise ValueError('x_lower should be less than x_upper.')
            return __cdf(self.mean_val, self.std_val, x_upper) - __cdf(self.mean, self.std_val, x_lower)
        return __cdf(self.mean_val, self.std_val, self.randvar)

    def confidence_interval(self) -> Union[int, float]:
        # find critical values for a given p-value
        pass

    def mean(self) -> Union[int, float]:
        """
        Returns: Mean of the Gaussian distribution
        """
        return self.mean_val

    def median(self) -> Union[int, float]:
        """
        Returns: Median of the Gaussian distribution
        """
        return self.mean_val

    def mode(self) -> Union[int, float]:
        """
        Returns: Mode of the Gaussian distribution
        """
        return self.mean_val

    def var(self) -> Union[int, float]:
        """
        Returns: Variance of the Gaussian distribution
        """
        return pow(self.std_val, 2)

    def std(self) -> Union[int, float]:
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
        return _log(self.std*_sqrt(2 * _pi * _e))

    def summary(self, display=False) -> Union[None, Tuple[str, str, str, str, str, str, str]]:
        """
        Returns:  summary statistic regarding the Gaussian distribution which contains the following parts of the distribution:
                (mean, median, mode, var, std, skewness, kurtosis). If the display parameter is True, the function returns None
                and prints out the summary of the distribution. 
        """
        if display == True:
            cstr = " summary statistics "
            print(cstr.center(40, "="))
            print(f"mean: {self.mean()}", f"median: {self.median()}",
                  f"mode: {self.mode()}", f"var: {self.var()}", f"std: {self.std()}",
                  f"skewness: {self.skewness()}", f"kurtosis: {self.kurtosis()}", sep='\n')

            return None
        else:
            return (f"mean: {self.mean()}", f"median: {self.median()}",
                    f"mode: {self.mode()}", f"var: {self.var()}", f"std: {self.std()}",
                    f"skewness: {self.skewness()}", f"kurtosis: {self.kurtosis()}")

    def keys(self) -> Dict[str, Union[float, int, str]]:
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

