try:
    import numpy as _np
    from math import sqrt as _sqrt, pi as _pi, exp as _exp
    from typing import Union, Tuple, Dict, List
    from _base import Infinite
except Exception as e:
    print(f"some modules are missing {e}")

class Logistic(Infinite):
    """
    This class contains methods concerning Logistic Distirbution.
    Args:

        location(float): mean parameter
        scale(float | x>0): standard deviation
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

    Reference:
    - Wikipedia contributors. (2020, December 12). Logistic distribution. In Wikipedia, The Free Encyclopedia.
     Retrieved 11:14, December 28, 2020, from https://en.wikipedia.org/w/index.php?title=Logistic_distribution&oldid=993793195
    """
    def __init__(self, location: float, scale: float, randvar: float):
        if scale < 0:
            raise ValueError(f'scale should be greater than 0. Entered value for Scale:{scale}')

        self.scale = scale
        self.location = location
        self.randvar = randvar

    def pdf(self, x: Union[List[float], _np.ndarray] = None) -> Union[float, _np.ndarray]:
        """
        Args:

            x (List[float], numpy.ndarray): random variable or list of random variables

        Returns:
            either probability density evaluation for some point or plot of Logistic distribution.
        """
        mu = self.location
        s = self.scale
        randvar = self.randvar

        __generator=lambda mu, s, x: _np.exp(-(x - mu) / s) / (s * (1 + _np.exp(
            -(x - mu) / s))**2)

        if x is not None:
            if not (isinstance(x, _np.ndarray)) and issubclass(x, List):
                raise TypeError(f'parameter x only accepts List types or numpy.ndarray')
            else:
                x = _np.array(x)
                return _np.exp(-(x - mu) / s) / (s * (1 + _np.power(_np.exp(-(x - mu) / s)), 2))
        return _exp(-(randvar - mu) / s) / (s * (1 + pow(_exp(-(randvar - mu) / s)), 2))

    def cdf(self, x: Union[List[float], _np.ndarray] = None) -> Union[float, _np.ndarray]:
        """
        Args:

            x (List[float], numpy.ndarray): random variable or list of random variables

        Returns:
            either cumulative distribution evaluation for some point or plot of Logistic distribution.
        """
        mu = self.location
        s = self.scale
        randvar = self.randvar

        if x is not None:
            if not (isinstance(x, _np.ndarray)) and issubclass(x, List):
                raise TypeError(f'parameter x only accepts List types or numpy.ndarray')
            else:
                x = _np.array(x)
                return 1 / (1 + _np.exp(-(x - mu) / s))
        return 1 / (1 + _exp(-(randvar - mu) / s))

    def pvalue(self, x_lower=-_np.inf, x_upper=None):
        """
        Args:

            x_lower(float): defaults to 0. Defines the lower value of the distribution. Optional.
            x_upper(float): defaults to None. If not defined defaults to random variable x. Optional.

            Note: definition of x_lower and x_upper are only relevant when probability is between two random variables.
            Otherwise, the default random variable is x.

        Returns:
            p-value of the Logistic distribution evaluated at some random variable.
        """
        if x_upper == None:
            x_upper=self.randvar
        if x_lower > x_upper:
            raise ValueError(f'lower bound should be less than upper bound. \
                            Entered values: x_lower:{x_lower} x_upper:{x_upper}')
        __cdf=lambda mu, s, x: 1 / (1 + _np.exp(-(x - mu) / s))
        return __cdf(self.location, self.scale, x_upper) - __cdf(self.location, self.scale, x_lower)

    def mean(self) -> float:
        """
        Returns: Mean of the Logistic distribution.
        """
        return self.location

    def median(self) -> float:
        """
        Returns: Median of the Logistic distribution.
        """
        return self.location

    def mode(self) -> float:
        """
        Returns: Mode of the Logistic distribution.
        """
        return self.location

    def var(self) -> float:
        """
        Returns: Variance of the Logistic distribution.
        """
        return pow(self.scale, 2) * pow(_pi, 2)/3

    def std(self) -> float:
        """
        Returns: Standard deviation of the Logistic distribution.
        """
        return _sqrt(self.var())

    def skewness(self) -> float:
        """
        Returns: Skewness of the Logistic distribution.
        """
        return 0.0

    def kurtosis(self) -> float:
        """
        Returns: Kurtosis of the Logistic distribution.
        """
        return 6 / 5

    def entropy(self) -> float:
        """
        Returns: differential entropy of the Logistic distribution.

        Reference: Park, S.Y. & Bera, A.K.(2009). Maximum entropy autoregressive conditional heteroskedasticity model. Elsivier.
        link: http://wise.xmu.edu.cn/uploadfiles/paper-masterdownload/2009519932327055475115776.pdf
        """
        return 2.0

    def summary(self, display=False) -> Union[None, Tuple[str, str, str, str, str, str, str]]:
        """
        Returns:  summary statistic regarding the Logistic distribution which contains the following parts of the distribution:
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
        Summary statistic regarding the Logistic distribution which contains the following parts of the distribution:
        (mean, median, mode, var, std, skewness, kurtosis).

        Returns:
            Dict[str, Union[float, int]]: [description]
        """
        return {
            'mean': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.var(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }

