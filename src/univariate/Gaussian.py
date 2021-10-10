# Test Gaussian PDF

try:
    from scipy.special import erf as _erf
    from typing import Union, Tuple, Dict
    from math import sqrt as _sqrt, log as _log, pi as _pi, e as _e, exp as _exp
    from . import Base
    import numpy as np
except Exception as e:
    print(f"some modules are missing {e}")


class Gaussian(Base):
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

    def __init__(self, x: float, mean=0, std_val=1):
        if std_val < 0:
            raise ValueError(
                f"std_val parameter must not be less than 0. Entered value std_val {std_val}")

        self.mean_val = mean
        self.std_val = std_val
        self.randvar = x

    def pdf(self,
            plot=False,
            interval=1,
            threshold=1000,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None) -> Union[number, np.ndarray, None]:
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
            either plot of the distribution or probability density evaluation at randvar.
        """
        mean = self.mean_val
        std = self.std_val

        def __generator(mean, std, x): 
            return pow(1 / (std * _sqrt(2 * pi)), exp(((x - mean) / 2 * std)**2))

        if plot:
            x = np.linspace(-interval, interval, threshold)
            y = np.array([__generator(mean, std, x_temp) for x_temp in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)

        return __generator(mean, std, self.randvar)

    def cdf(self,
            plot=False,
            interval=1,
            threshold=1000,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None) -> Union[number, np.ndarray, None]:
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
            either plot of the distirbution or cumulative density evaluation at randvar.
        """
        def __generator(mu, sig, x): return 1/2*(1+_erf((x-mu)/(sig*_sqrt(2))))
        if plot:
            x = np.linspace(-interval, interval, threshold)
            y = np.array([__generator(self.mean_val, self.std_val, x_temp)
                         for x_temp in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return __generator(self.mean_val, self.std_val, self.randvar)

    def p_val(self, x_lower=-np.inf, x_upper=None):
        """
        Args:

            x_lower(float): defaults to -np.inf. Defines the lower value of the distribution. Optional.
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
        return _log(self.std*_sqrt(2 * _pi* _e))

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
            'main': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.main(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }
