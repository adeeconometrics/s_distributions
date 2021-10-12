try:
    from scipy.special import erfc as _erfc
    import numpy as np
    from math import sqrt as _sqrt, log as _log, pi as _pi, exp as _exp
    from typing import Union, Tuple, Dict
    from . import _base
except Exception as e:
    print(f"some modules are missing {e}")

class LogNormal(Base):
    """
    This class contains methods concerning the Log Normal Distribution.

    Args:

        randvar(float | [0, infty)): random variable
        mean_val(float): mean parameter
        std_val(float | x>0): standard deviation

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

    References:
    - Weisstein, Eric W. "Log Normal Distribution." From MathWorld--A Wolfram Web Resource.
    https://mathworld.wolfram.com/LogNormalDistribution.html
    - Wikipedia contributors. (2020, December 18). Log-normal distribution. In Wikipedia, The Free Encyclopedia.
    Retrieved 06:49, December 23, 2020, from https://en.wikipedia.org/w/index.php?title=Log-normal_distribution&oldid=994919804
    """
    def __init__(self, mean: float, std_val: float, randvar: float):
        if randvar < 0:
            raise ValueError(
                f'random variable should be greater than 0. Entered value for randvar:{randvar}')
        if std < 0:
            raise ValueError(
                f'random variable should be greater than 0. Entered value for std:{std}')
        self.randvar = randvar
        self.mean_val = mean
        self.std_val = std_val

    def pdf(self,
            plot=False,
            interval=1,
            threshold=1000,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None) -> Union[float, int, np.ndarray, None]:
        """
        Args:

            interval(int): defaults to none. Only necessary for defining plot.
            threshold(int): defaults to 1000. Defines the sample points in plot.
            plot(bool): if true, returns plot.
            xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
            ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true.
            xlabel(string): sets label in x axis. Only relevant when plot is true.
            ylabel(string): sets label in y axis. Only relevant when plot is true.


        Returns:
            either probability density evaluation for some point or plot of Log Normal-distribution.
        """
        __generator = lambda mean, std, x: (1 / (x * std * _sqrt(
            2 * _pi))) * _exp(-(_log(x - mean)**2) / (2 * std**2))

        if plot:
            x=np.linspace(-interval, interval, int(threshold))
            y=np.array([__generator(self.mean_val, self.std_val, i)
                       for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return __generator(self.mean_val, self.std_val, self.randvar)

    def cdf(self,
            plot=False,
            interval=1,
            threshold=1000,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None) -> Union[float, int, np.ndarray, None]:
        """
        Args:

            interval(int): defaults to none. Only necessary for defining plot.
            threshold(int): defaults to 1000. Defines the sample points in plot.
            plot(bool): if true, returns plot.
            xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
            ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true.
            xlabel(string): sets label in x axis. Only relevant when plot is true.
            ylabel(string): sets label in y axis. Only relevant when plot is true.


        Returns:
            either cumulative distribution evaluation for some point or plot of Log Normal-distribution.
        """
        __generator = lambda mean, std, x: 0.5 + 0.5*_erfc(-(_log(x - mean) /
                                                           (std * _sqrt(2))))
        if plot:
            x=np.linspace(-interval, interval, int(threshold))
            y=np.array([__generator(self.mean_val, self.std_val, i)
                       for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return __generator(self.mean_val, self.std_val, self.randvar)

    def pvalue(self, x_lower=0, x_upper=None) -> Optional[float, int]:
        """
        Args:

            x_lower(float): defaults to 0. Defines the lower value of the distribution. Optional.
            x_upper(float): defaults to None. If not defined defaults to random variable x. Optional.

            Note: definition of x_lower and x_upper are only relevant when probability is between two random variables.
            Otherwise, the default random variable is x.

        Returns:
            p-value of the Log Normal-distribution evaluated at some random variable.
        """
        __cdf=lambda mean, std, x: 0.5 + 0.5* _erfc(-( _log(x - mean) /
                                                           (std * _sqrt(2))))
        if x_lower < 0:
            raise ValueError(f'x_lower should not be less then 0. X_lower: {x_lower}')
        if x_upper == None:
            x_upper=self.randvar

        return __cdf(self.mean_val, self.std_val, x_upper) - __cdf(self.mean_val, self.std_val, x_lower)

    def mean(self) -> Union[float, int]:
        """
        Returns: Mean of the log normal distribution.
        """
        return _exp(self.mean_val + pow(self.std_val, 2) / 2)

    def median(self) -> Union[float, int]:
        """
        Returns: Median of the log normal distribution.
        """
        return _exp(self.mean_val)

    def mode(self) -> Union[float, int]:
        """
        Returns: Mode of the log normal distribution.
        """
        return _exp(self.mean_val - pow(self.std_val, 2))

    def var(self) -> Union[float, int]:
        """
        Returns: Variance of the log normal distribution.
        """
        std=self.std_val
        mean=self.mean_val
        return (_exp(pow(std, 2)) - 1) * _exp(2 * mean + pow(std, 2))

    def std(self) -> Union[float, int]:
        """
        Returns: Standard deviation of the log normal distribution
        """
        return self.std_val

    def skewness(self) -> Union[float, int]:
        """
        Returns: Skewness of the log normal distribution.
        """
        std=self.std_val
        mean=self.mean_val
        return (_exp(pow(std, 2)) + 2) * _sqrt(_exp(pow(std, 2)) - 1)

    def kurtosis(self) -> Union[float, int]:
        """
        Returns: Kurtosis of the log normal distribution.
        """
        std=self.std_val
        return _exp(
            4 * pow(std, 2)) + 2 * _exp(3 * pow(std, 2)) + 3 * _exp(2 * pow(std, 2)) - 6

    def entropy(self) -> Union[float, int]:
        """
        Returns: differential entropy of the log normal distribution.

        Reference: Park, S.Y. & Bera, A.K.(2009). Maximum entropy autoregressive conditional heteroskedasticity model. Elsivier.
        link: http://wise.xmu.edu.cn/uploadfiles/paper-masterdownload/2009519932327055475115776.pdf
        """
        return self.mean_val + 0.5 *_log(2*_pi*_e*self.std_val**2)

    def summary(self, display=False) -> Union[None, Tuple[str, str, str, str, str, str, str]]:
        """
        Returns:  summary statistic regarding the LogNormal distribution which contains the following parts of the distribution:
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
        Summary statistic regarding the LogNormal distribution which contains the following parts of the distribution:
        (mean, median, mode, var, std, skewness, kurtosis).

        Returns:
            Dict[str, Union[float, int]]: [description]
        """
        return {
            'main': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.main(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }
