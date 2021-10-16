try:
    from numpy import euler_gamma as _euler_gamma
    import numpy as np
    from math import sqrt as _sqrt, log as _log, pi as _pi
    from typing import Union, Tuple, Dict
    from _base import Base
except Exception as e:
    print(f"some modules are missing {e}")


class Gumbel(Base):
    """
    This class contains methods concerning Gumbel Distirbution.
    Args:

        location(float): location parameter
        scale(float>0): scale parameter
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
    - Wikipedia contributors. (2020, November 26). Gumbel distribution. In Wikipedia, The Free Encyclopedia.
    Retrieved 09:22, December 29, 2020, from https://en.wikipedia.org/w/index.php?title=Gumbel_distribution&oldid=990718796
    """

    def __init__(self, location: Union[float, int], scale: Union[float, int], randvar: Union[float, int]):
        if scale < 0:
            raise ValueError(
                'scale parameter should be greater than 0. The value of the scale parameter is: {scale}')

        self.location = location
        self.scale = scale
        self.randvar = randvar

    def pdf(self,
            plot=False,
            interval=1,
            threshold=1000,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None) -> Union[float, np.ndarray, None]:
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
            either probability density evaluation for some point or plot of Gumbel distribution.
        """
        def __generator(mu, beta, x):
            z = (x-mu)/beta
            return (1/beta)*np.exp(-(z+np.exp(-z)))

        if plot:
            x = np.linspace(-interval, interval, int(threshold))
            y = np.array([__generator(self.location, self.scale, i)
                         for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return __generator(self.location, self.scale, self.randvar)

    def cdf(self,
            plot=False,
            interval=1,
            threshold=1000,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None) -> Union[float, np.ndarray, None]:
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
            either cumulative distribution evaluation for some point or plot of Gumbel distribution.
        """
        def __generator(mu, beta, x):
            return np.exp(-np.exp(-(x-mu)/beta))
        if plot:
            x = np.linspace(-interval, interval, int(threshold))
            y = np.array([__generator(self.location, self.scale, i)
                         for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return __generator(self.location, self.scale, self.randvar)

    def pvalue(self) -> float:
        """
        Args:

            x_lower(float): defaults to 0. Defines the lower value of the distribution. Optional.
            x_upper(float): defaults to None. If not defined defaults to random variable x. Optional.

            Note: definition of x_lower and x_upper are only relevant when probability is between two random variables.
            Otherwise, the default random variable is x.

        Returns:
            p-value of the Gumbel distribution evaluated at some random variable.
        """
        return "currently unsupported"

    def mean(self) -> float:
        """
        Returns: Mean of the Gumbel distribution.
        """
        return self.location+(self.scale*_euler_gamma)

    def median(self) -> float:
        """
        Returns: Median of the Gumbel distribution.
        """
        return self.location - (self.scale*_log(_log(2)))

    def mode(self) -> float:
        """
        Returns: Mode of the Gumbel distribution.
        """
        return self.location

    def var(self) -> float:
        """
        Returns: Variance of the Gumbel distribution.
        """
        return pow(_pi, 2/6)*pow(self.scale, 2)

    def std(self) -> float:
        """
        Returns: Standard deviation of the Gumbel distribution.
        """
        return _sqrt(self.var())

    def skewness(self) -> float:
        """
        Returns: Skewness of the Gumbel distribution.
        """
        return 1.14

    def kurtosis(self) -> float:
        """
        Returns: Kurtosis of the Gumbel distribution.
        """
        return 12/5

    def summary(self, display=False) -> Union[None, Tuple[str, str, str, str, str, str, str]]:
        """
        Returns:  summary statistic regarding the Gumbell distribution which contains the following parts of the distribution:
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

    def keys(self) -> Dict[str, float]:
        """
        Summary statistic regarding the Gumbell distribution which contains the following parts of the distribution:
        (mean, median, mode, var, std, skewness, kurtosis).

        Returns:
            Dict[str, float]: [description]
        """
        return {
            'mean': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.var(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }

