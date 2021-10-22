try:
    import numpy as _np
    from typing import Union, Tuple, Dict, List
    from math import sqrt as _sqrt, pi as _pi, asin as _asin
    from univariate._base import BoundedInterval
except Exception as e:
    print(f"some modules are missing {e}")


class Arcsine(BoundedInterval):
    """
    This class contains methods concerning Arcsine Distirbution.
    Args:

        randvar(float in [0, 1]): random variable

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

    Reference:
    - Wikipedia contributors. (2020, October 30). Arcsine distribution. In Wikipedia, The Free Encyclopedia.
    Retrieved 05:19, December 30, 2020, from https://en.wikipedia.org/w/index.php?title=Arcsine_distribution&oldid=986131091
    """

    def __init__(self, randvar: Union[float, int]):
        if randvar < 0 or randvar > 1:
            raise ValueError(
                f'random variable should have values between [0,1]. The value of randvar was: {randvar}')

        self.randvar = randvar

    def pdf(self, x: Union[List[float], _np.ndarray, float] = None) -> Union[float, _np.ndarray]:
        """
        Args:

            x (List[float], numpy.ndarray): random variable or list of random variables

        Returns:
            either probability density evaluation for some point or plot of Arcsine distribution.
        """

        if x is not None:
            if not isinstance(x, (_np.ndarray, List)):
                raise TypeError('parameter x only accepts List types or numpy.ndarray')
            else:
                x = _np.array(x)
                return 1/(_pi * _np.sqrt(x*(1-x)))

        return 1/_pi*_sqrt(self.randvar * (1-self.randvar))

    def cdf(self, x: Union[List[float], numpy.ndarray] = None) -> Union[float, numpy.ndarray]:
        """
        Args:

            x (List[float], numpy.ndarray): random variable or list of random variables

        Returns:
            either cumulative distribution evaluation for some point or plot of Arcsine distribution.
        """

        if x is not None:
            if not isinstance(x, (_np.ndarray, List)):
                raise TypeError('parameter x only accepts List types or numpy.ndarray')
            else:
                x = _np.array(x)
                return 1/(_pi)*_np.arcsin(_np.sqrt(x))

        return 1/_pi * _asin(_sqrt(self.randvar))

    def pvalue(self, x_lower=0, x_upper=None) -> Optional[float]:
        """
        Args:

            x_lower(float): defaults to 0. Defines the lower value of the distribution. Optional.
            x_upper(float): defaults to None. If not defined defaults to random variable x. Optional.

            Note: definition of x_lower and x_upper are only relevant when probability is between two random variables.
            Otherwise, the default random variable is x.

        Returns:
            p-value of the Arcsine distribution evaluated at some random variable.
        """
        if x_lower < 0 or x_lower > 1:
            raise ValueError(
                f'x_lower should only be in between 0 and 1. X_lower:{x_lower}')
        if x_upper == None:
            x_upper = self.randvar
        if x_lower > x_upper:
            raise ValueError(
                f'lower bound should be less than upper bound. Entered values: x_lower:{x_lower} x_upper:{x_upper}')

        def __cdf(x): return (2/_pi)*_asin(_sqrt(x))
        return __cdf(x_upper)-__cdf(x_lower)

    def mean(self) -> float:
        """
        Returns: Mean of the Arcsine distribution.
        """
        return 0.5

    def median(self) -> float:
        """
        Returns: Median of the Arcsine distribution.
        """
        return 0.5

    def mode(self) -> Tuple[float, float]:
        """
        Returns: Mode of the Arcsine distribution. Mode is within the set {0,1}
        """
        return (0, 1)

    def var(self) -> float:
        """
        Returns: Variance of the Arcsine distribution.
        """
        return 0.125

    def std(self) -> float:
        """
        Returns: Standard deviation of the Arcsine distribution.
        """
        return _sqrt(0.125)

    def skewness(self) -> float:
        """
        Returns: Skewness of the Arcsine distribution.
        """
        return 0.0

    def kurtosis(self) -> float:
        """
        Returns: Kurtosis of the Arcsine distribution.
        """
        return 1.5

    def summary(self, display=False) -> Union[None, Tuple[str, str, str, str, str, str, str]]:
        """
        Returns:  summary statistic regarding the Arcsine distribution which contains the following parts of the distribution:
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

    def keys(self) -> Dict[str, Union[float, Tuple[float]]]:
        """
        Summary statistic regarding the Arcsine distribution which contains the following parts of the distribution:
        (mean, median, mode, var, std, skewness, kurtosis).

        Returns:
            Dict[str, Union[float, Tuple[float]]]: [description]
        """
        return {
            'mean': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.var(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }
