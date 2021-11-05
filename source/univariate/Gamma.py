try:
    from scipy.special import gamma as _gamma, gammainc as _gammainc, digamma as _digamma
    import numpy as _np
    from math import sqrt as _sqrt, log as _log, exp as _exp
    from typing import Union, Tuple, Dict, List
    from univariate._base import SemiInfinite
except Exception as e:
    print(f"some modules are missing {e}")


class Gamma(SemiInfinite):
    """
    This class contains methods concerning a variant of Gamma distribution.

    Args:

        a(float | [0, infty)): shape
        b(float | [0, infty)): scale
        x(float | [0, infty)): random variable

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
    - Matlab(2020). Gamma Distribution.
    Retrieved from: https://www.mathworks.com/help/stats/gamma-distribution.html?searchHighlight=gamma%20distribution&s_tid=srchtitle
    """

    def __init__(self, a: float, b: float, x: float):
        if a < 0:
            raise ValueError(
                f'shape should be greater than 0. Entered value for a:{a}')
        if b < 0:
            raise ValueError(
                f'scale should be greater than 0. Entered value for b:{b}')
        if x < 0:
            raise ValueError(
                f'random variable should be greater than 0. Entered value for x:{b}')
        self.a = a
        self.b = b
        self.x = x

    def pdf(self, x: Union[List[float], _np.ndarray] = None) -> Union[float, _np.ndarray]:
        """
        Args:

            x (List[float], numpy.ndarray): random variable or list of random variables

        Returns:
            either probability density evaluation for some point or plot of Gamma distribution.
        """
        a = self.a
        b = self.b
        randvar = self.x

        if x is not None:
            if not isinstance(x, (_np.ndarray,List)):
                raise TypeError(
                    f'parameter x only accepts List types or numpy.ndarray')
            else:
                x = _np.array(x)
                return (1 / (pow(b, a) * _gamma(a))) * _np.log(x, a - 1) * _np.exp(-x / b)
        return (1 / (pow(b, a) * _gamma(a))) * _log(randvar, a - 1) * _exp(-randvar / b)

    def cdf(self, x: Union[List[float], _np.ndarray] = None) -> Union[float, _np.ndarray]:
        """
        Args:

            x (List[float], numpy.ndarray): random variable or list of random variables

        Returns:
            either cumulative distribution evaluation for some point or plot of Gamma distribution.
        """
        a = self.a
        b = self.b
        randvar = self.x

        # there is no apparent explanation for reversing gammainc's parameter, but it works quite perfectly in my prototype
        def __generator(a: float, b: float, x: Union[float, _np.ndarray]) -> Union[float, _np.ndarray]:
            return 1 - _gammainc(a, x / b)

        if x is not None:
            if not isinstance(x, (_np.ndarray,List)):
                raise TypeError(
                    f'parameter x only accepts List types or numpy.ndarray')
            else:
                x = _np.array(x)
                return __generator(a, b, x)
        return __generator(a, b, randvar)

    def pvalue(self, x_lower=0, x_upper=None) -> Union[float, int]:
        """
        Args:

            x_lower(float): defaults to 0. Defines the lower value of the distribution. Optional.
            x_upper(float): defaults to None. If not defined defaults to random variable x. Optional.

            Note: definition of x_lower and x_upper are only relevant when probability is between two random variables.
            Otherwise, the default random variable is x.

        Returns:
            p-value of the Gamma distribution evaluated at some random variable.
        """
        if x_lower < 0:
            raise ValueError(
                f'x_lower cannot be lower than 0. Entered value: {x_lower}')
        if x_upper is None:
            x_upper = self.x

        def __cdf(a, b, x): return 1 - _gammainc(a, x / b)

        return __cdf(self.a, self.b, x_upper, self.lambda_) - __cdf(self.a, self.b, x_lower, self.lambda_)

    def mean(self) -> Union[float, int]:
        """
        Returns: Mean of the Gamma distribution
        """
        return self.a * self.b

    def median(self) -> str:
        """
        Returns: Median of the Gamma distribution.
        """
        return "No simple closed form."

    def mode(self) -> Union[float, int]:
        """
        Returns: Mode of the Gamma distribution
        """
        return (self.a - 1) * self.b

    def var(self) -> Union[float, int]:
        """
        Returns: Variance of the Gamma distribution
        """
        return self.a * pow(self.b, 2)

    def std(self) -> float:
        """
        Returns: Standard deviation of the Gamma distribution
        """
        return _sqrt(self.var())

    def skewness(self) -> float:
        """
        Returns: Skewness of the Gamma distribution
        """
        return 2 / _sqrt(self.a)

    def kurtosis(self) -> float:
        """
        Returns: Kurtosis of the Gamma distribution
        """
        return 6 / self.a

    def entropy(self) -> float:
        """
        Returns: differential entropy of the Gamma distribution

        Reference: Park, S.Y. & Bera, A.K.(2009). Maximum entropy autoregressive conditional heteroskedasticity model. Elsivier.
        link: http://wise.xmu.edu.cn/uploadfiles/paper-masterdownload/2009519932327055475115776.pdf
        """
        k = self.a
        theta = self.b
        return k + _log(theta)+_log(_gamma(k))-(1-k)*_digamma(k)

    def summary(self, display=False) -> Union[None, Tuple[str, str, str, str, str, str, str]]:
        """
        Returns:  summary statistic regarding the Gamma distribution which contains the following parts of the distribution:
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
        Summary statistic regarding the Gamma distribution which contains the following parts of the distribution:
        (mean, median, mode, var, std, skewness, kurtosis).

        Returns:
            Dict[str, Union[float, int, str]]: [description]
        """
        return {
            'mean': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.var(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }
