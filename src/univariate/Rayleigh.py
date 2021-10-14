try:
    import numpy as np
    from numpy import euler_gamma as _euler_gamma
    from typing import Union, Tuple, Dict
    from math import sqrt as _sqrt, log as _log, pi as _pi
    from _base import Base
except Exception as e:
    print(f"some modules are missing {e}")


class Rayleigh(Base):
    """
    This class contains methods concerning Rayleigh Distirbution.
    Args:

        scale(float | x>0): scale
        randvar(float | x>=0): random variable

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

    Reference:
    - Wikipedia contributors. (2020, December 30). Rayleigh distribution. In Wikipedia, The Free Encyclopedia.
    Retrieved 09:37, January 8, 2021, from https://en.wikipedia.org/w/index.php?title=Rayleigh_distribution&oldid=997166230

    - Weisstein, Eric W. "Rayleigh Distribution." From MathWorld--A Wolfram Web Resource.
    https://mathworld.wolfram.com/RayleighDistribution.html
    """

    def __init__(self, scale: float, randvar: float):
        if randvar < 0:
            raise ValueError(
                'random variable should be a positive number. Entered value: {}'.format(randvar))
        if scale < 0:
            raise ValueError('scale parameter should be a positive number.')

        self.scale = scale
        self.randvar = randvar

    def pdf(self,
            plot=False,
            interval=0,
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
            either probability density evaluation for some point or plot of Rayleigh distribution.
        """
        def __generator(sig, x): return (x/pow(sig, 2)) * \
            np.exp(pow(-x, 2)/(2*pow(sig, 2)))

        if plot:
            if interval < 0:
                raise ValueError(
                    'interval should not be less then 0. Entered value: {}'.format(interval))
            x = np.linspace(0, interval, int(threshold))
            y = np.array([__generator(self.scale, i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return __generator(self.scale, self.randvar)

    def cdf(self,
            plot=False,
            threshold=1000,
            interval=1,
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
            either cumulative distribution evaluation for some point or plot of Rayleigh distribution.
        """
        def __generator(sig, x): return 1-np.exp(-x**2/(2*sig**2))
        if plot:
            x = np.linspace(0, interval, int(threshold))
            y = np.array([__generator(self.scale, i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return __generator(self.scale, self.randvar)

    def pvalue(self, x_lower=0, x_upper=None) -> Optional[float]:
        """
        Args:

            x_lower(float): defaults to 0. Defines the lower value of the distribution. Optional.
            x_upper(float): defaults to None. If not defined defaults to random variable x. Optional.

            Note: definition of x_lower and x_upper are only relevant when probability is between two random variables.
            Otherwise, the default random variable is x.

        Returns:
            p-value of the Rayleigh distribution evaluated at some random variable.
        """
        if x_upper == None:
            x_upper = self.randvar
        if x_lower > x_upper:
            raise Exception(
                'lower bound should be less than upper bound. Entered values: x_lower:{} x_upper:{}'.format(x_lower, x_upper))

        def __cdf(sig, x): return 1-np.exp(-x**2/(2*sig**2))
        return __cdf(self.scale, x_upper)-__cdf(self.scale, x_lower)

    def mean(self) -> float:
        """
        Returns: Mean of the Rayleigh distribution.
        """
        return self.scale*_sqrt(_pi/2)

    def median(self) -> float:
        """
        Returns: Median of the Rayleigh distribution.
        """
        return self.scale*_sqrt(2*_log(2))

    def mode(self) -> float:
        """
        Returns: Mode of the Rayleigh distribution.
        """
        return self.scale

    def var(self) -> float:
        """
        Returns: Variance of the Rayleigh distribution.
        """
        return (4-_pi)/2*pow(self.scale, 2)

    def std(self) -> float:
        """
        Returns: Standard deviation of the Rayleigh distribution
        """
        return _sqrt((4-_pi)/2*pow(self.scale, 2))

    def skewness(self) -> float:
        """
        Returns: Skewness of the Rayleigh distribution.
        """
        return (2*_sqrt(_pi)*(_pi-3))/pow((4-_pi), 3/2)

    def kurtosis(self) -> float:
        """
        Returns: Kurtosis of the Rayleigh distribution.
        """
        return -(6*pow(_pi, 2)-24*_pi+16)/pow(4-_pi, *2)

    def entropy(self) -> float:
        """
        Returns: differential entropy of the Rayleigh distribution.

        Reference: Park, S.Y. & Bera, A.K.(2009). Maximum entropy autoregressive conditional heteroskedasticity model. Elsivier.
        link: http://wise.xmu.edu.cn/uploadfiles/paper-masterdownload/2009519932327055475115776.pdf
        """
        return 1+_log(self.scale/_sqrt(2))+(_euler_gamma/2)

    def summary(self, display=False) -> Union[None, Tuple[str, str, str, str, str, str, str]]:
        """
        Returns:  summary statistic regarding the Rayleigh distribution which contains the following parts of the distribution:
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
        Summary statistic regarding the Rayleigh distribution which contains the following parts of the distribution:
        (mean, median, mode, var, std, skewness, kurtosis).

        Returns:
            Dict[str, Union[float, Tuple[float]]]: [description]
        """
        return {
            'main': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.main(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }
