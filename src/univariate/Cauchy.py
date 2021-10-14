try:
    import numpy as np
    from math import log as _log, log10 as _log10, pi as _pi
    from typing import Union, Tuple, Dict
    from _base import Base
except Exception as e:
    print(f"some modules are missing {e}")


class Cauchy(Base):
    """
    This class contains methods concerning the Cauchy Distribution.

    Args:

        scale(float | x>0): pertains to  the scale parameter
        location(float): pertains to the location parameter or median
        x(float): random variable

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
    - Wikipedia contributors. (2020, November 29). Cauchy distribution. In Wikipedia, The Free Encyclopedia.
    Retrieved 12:01, December 22, 2020, from https://en.wikipedia.org/w/index.php?title=Cauchy_distribution&oldid=991234690
    - Weisstein, Eric W. "Cauchy Distribution." From MathWorld--A Wolfram Web Resource. https://mathworld.wolfram.com/CauchyDistribution.html
    """

    def __init__(self, x: Union[float, int], location: Union[float, int], scale: Union[float, int]):
        # if (type(x) and type(location) and type(scale)) not in (int, float):
        #     raise TypeError('arguments must be of type int or float.')
        if scale < 0:
            raise ValueError(
                f'scale should be greater than 0. Entered value for scale:{scale}')
        self.scale = scale
        self.location = location
        self.x = x

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
            either probability density evaluation for some point or plot of Cauchy distribution.
        """
        x = self.x
        location = self.location
        scale = self.scale

        def __generator(x, location, scale):
            return 1 / (pi * scale * (1 + pow((x - location) / scale, 2))

        if plot:
            x=np.linspace(-interval, interval, int(threshold))
            y=np.array([__generator(i, location, scale) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)

        return __generator(x, location, scale)

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
            either cumulative distirbution evaluation for some point or plot of Cauchy distribution.
        """
        x=self.x
        location=self.location
        scale=self.scale
        def __generator(x, location, scale): return (1 / _pi) * np.arctan(
            (x - location) / scale) + 1 / 2
        if plot:
            x=np.linspace(-interval, interval, int(threshold))
            y=np.array([__generator(i, location, scale) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)

        return __generator(x, location, scale)

    def pvalue(self, x_lower=-np.inf, x_upper=None):
        """
        Args:

            x_lower(float): defaults to -np.inf. Defines the lower value of the distribution. Optional.
            x_upper(float | x_upper>x_lower): defaults to None. Defines the upper value of the distribution. Optional.

            Note: definition of x_lower and x_upper are only relevant when probability is between two random variables.
            Otherwise, the default random variable is x.

        Returns:
            p-value of the Cauchy distribution evaluated at some random variable.
        """
        def __cdf(x, location, scale): return (
            1 / _pi) * np.arctan((x - location) / scale) + 1 / 2
        if x_upper != None:
            if x_lower > x_upper:
                raise ValueError('x_lower should be less than x_upper.')
            return __cdf(x_upper, self.location, self.scale) - __cdf(x_lower, self.location, self.scale)
        return __cdf(self.x, self.location, self.scale)

    def confidence_interval(self) -> Union[float, str]:
        pass

    def mean(self) -> str:
        """
        Returns: Mean of the Cauchy distribution. Mean is Undefined.
        """
        return "undefined"

    def median(self) -> float:
        """
        Returns: Median of the Cauchy distribution.
        """
        return self.location

    def mode(self) -> float:
        """
        Returns: Mode of the Cauchy distribution
        """
        return self.location

    def var(self) -> str:
        """
        Returns: Variance of the Cauchy distribution.
        """
        return "undefined"

    def std(self) -> str:
        """
        Returns: Standard Deviation of the Cauchy Distribution.
        """
        return "undefined"

    def skewness(self) -> str:
        """
        Returns: Skewness of the Cauchy distribution.
        """
        return "undefined"

    def kurtosis(self) -> float:
        """
        Returns: Kurtosis of the Cauchy distribution
        """
        return _log(4 * _pi * self.scale)

    def entropy(self) -> float:
        """
        Returns: differential entropy of the Cauchy distribution

        Reference: Park, S.Y. & Bera, A.K.(2009). Maximum entropy autoregressive conditional heteroskedasticity model. Elsivier.
        link: http://wise.xmu.edu.cn/uploadfiles/paper-masterdownload/2009519932327055475115776.pdf
        """
        return _log10(4*_pi*self.scale)

    def summary(self, display=False) -> Union[None, Tuple[str, str, str, str, str, str, str]]:
        """
        Returns:  summary statistic regarding the ChiSquare-distribution which contains the following parts of the distribution:
                (mean, median, mode, var, std, skewness, kurtosis). If the display parameter is True, the function returns None
                and prints out the summary of the distribution.
        """
        if display == True:
            cstr=" summary statistics "
            print(cstr.center(40, "="))
            print(f"mean: {self.mean()}", f"median: {self.median()}",
                  f"mode: {self.mode()}", f"var: {self.var()}", f"std: {self.std()}",
                  f"skewness: {self.skewness()}", f"kurtosis: {self.kurtosis()}", sep='\n')

            return None
        else:
            return (f"mean: {self.mean()}", f"median: {self.median()}",
                    f"mode: {self.mode()}", f"var: {self.var()}", f"std: {self.std()}",
                    f"skewness: {self.skewness()}", f"kurtosis: {self.kurtosis()}")

    def keys(self) -> Dict[str, Union[float, str]]:
        """
        Summary statistic regarding the ChiSquare-distribution which contains the following parts of the distribution:
        (mean, median, mode, var, std, skewness, kurtosis).

        Returns:
            Dict[str, Union[float, str]]: [description]
        """
        return {
            'main': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.main(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }
