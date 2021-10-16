try:
    import numpy as np
    from math import sqrt as _sqrt, log as _log
    from typing import Union, Tuple, Dict
    from _base import Infinite
except Exception as e:
    print(f"some modules are missing {e}")

class Laplace(Infinite):
    """
    This class contains methods concerning Laplace Distirbution.
    Args:

        location(float): mean parameter
        scale(float| x>0): standard deviation
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
        - Wikipedia contributors. (2020, December 21). Laplace distribution. In Wikipedia, The Free Encyclopedia.
        Retrieved 10:53, December 28, 2020, from https://en.wikipedia.org/w/index.php?title=Laplace_distribution&oldid=995563221
    """
    def __init__(self, location: float, scale: float, randvar: float):
        if scale < 0:
            raise ValueError(f'scale should be greater than 0. Entered value for Scale:{scale}')

        self.scale=scale
        self.location=location
        self.randvar=randvar

    def pdf(self,
            plot=False,
            interval=1,
            threshold=1000,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None) -> Union[int, float, np.ndarray, None]:
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
            either probability density evaluation for some point or plot of Laplace distribution.
        """
        __generator=lambda mu, b, x: (1 / (2 * b)) * np.exp(abs(x - mu) / b)
        if plot:
            x=np.linspace(-interval, interval, int(threshold))
            y=np.array([__generator(self.location, self.scale, i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return __generator(self.location, self.scale, self.randvar)

    def cdf(self,
            plot=False,
            interval=1,
            threshold=1000,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None) -> Union[int, float, np.ndarray, None]:
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
            either cumulative distribution evaluation for some point or plot of Laplace distribution.
        """
        __generator=lambda mu, b, x: 1 / 2 + ((1 / 2) * np.sign(x - mu) *
                                              (1 - np.exp(abs(x - mu) / b)))
        if plot:
            x=np.linspace(-interval, interval, int(threshold))
            y=np.array([__generator(self.location, self.scale, i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return __generator(self.location, self.scale, self.randvar)

    def pvalue(self, x_lower=-np.inf, x_upper=None):
        """
        Args:

            x_lower(float): defaults to 0. Defines the lower value of the distribution. Optional.
            x_upper(float): defaults to None. If not defined defaults to random variable x. Optional.

            Note: definition of x_lower and x_upper are only relevant when probability is between two random variables.
            Otherwise, the default random variable is x.

        Returns:
            p-value of the Laplace distribution evaluated at some random variable.
        """
        if x_upper == None:
            x_upper=self.randvar
        if x_lower > x_upper:
            raise Exception(
                'lower bound should be less than upper bound. Entered values: x_lower:{} x_upper:{}'.format(x_lower, x_upper))
        __cdf=lambda mu, b, x: 1 / 2 + \
            ((1 / 2) * np.sign(x - mu) * (1 - np.exp(abs(x - mu) / b)))

        return __cdf(self.location, self.scale, x_upper)-__cdf(self.location, self.scale, x_lower)

    def mean(self) -> float:
        """
        Returns: Mean of the Laplace distribution.
        """
        return self.location

    def median(self) -> float:
        """
        Returns: Median of the Laplace distribution.
        """
        return self.location

    def mode(self) -> float:
        """
        Returns: Mode of the Laplace distribution.
        """
        return self.location

    def var(self) -> Union[int, float]:
        """
        Returns: Variance of the Laplace distribution.
        """
        return 2 * pow(self.scale, 2)

    def std(self) -> float:
        """
        Returns: Standard deviation of the Laplace distribution
        """
        return _sqrt(self.var())

    def skewness(self) -> float:
        """
        Returns: Skewness of the Laplace distribution.
        """
        return 0.0

    def kurtosis(self) -> float:
        """
        Returns: Kurtosis of the Laplace distribution.
        """
        return 3.0

    def entropy(self) -> float:
        """
        Returns: differential entropy of the Laplace distribution.

        Reference: Park, S.Y. & Bera, A.K.(2009). Maximum entropy autoregressive conditional heteroskedasticity model. Elsivier.
        link: http://wise.xmu.edu.cn/uploadfiles/paper-masterdownload/2009519932327055475115776.pdf
        """
        return 1 + _log(2*self.scale)

    def summary(self, display=False) -> Union[None, Tuple[str, str, str, str, str, str, str]]:
        """
        Returns:  summary statistic regarding the Laplace distribution which contains the following parts of the distribution:
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
        Summary statistic regarding the Laplace distribution which contains the following parts of the distribution:
        (mean, median, mode, var, std, skewness, kurtosis).

        Returns:
            Dict[str, Union[float, int]]: [description]
        """
        return {
            'mean': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.var(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }

