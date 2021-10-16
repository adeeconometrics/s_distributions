try:
    from scipy.special import gamma as _gamma
    import numpy as np
    from math import sqrt as _sqrt, log as _log, exp as _exp
    from typing import Union, Tuple, Dict
    from _base import SemiInfinite
except Exception as e:
    print(f"some modules are missing {e}")


class WeilbullInverse(SemiInfinite):
    """
    This class contains methods concerning inverse Weilbull or the Fréchet Distirbution.
    Args:

        shape(float | [0, infty)): shape parameter
        scale(float | [0,infty)]): scale parameter
        location(float | (-infty, infty)): location parameter
        randvar(float | randvar > location): random variable

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
    - Wikipedia contributors. (2020, December 7). Fréchet distribution. In Wikipedia, The Free Encyclopedia.
    Retrieved 07:28, December 30, 2020, from https://en.wikipedia.org/w/index.php?title=Fr%C3%A9chet_distribution&oldid=992938143
    """

    def __init__(self,  shape: Union[float, int], scale: Union[float, int], location: Union[float, int], randvar: Union[float, int]):
        if shape < 0 or scale < 0:
            raise ValueError(
                f'the value of scale and shape should be greater than 0. Entered values scale was:{scale}, shape:{shape}')
        if randvar < location:
            raise ValueError(
                f'random variable should be greater than the location parameter. Entered values: randvar: {randvar}, location:{location}')
        self.shape = shape
        self.scale = scale
        self.location = location
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
            either probability density evaluation for some point or plot of Fréchet distribution.
        """
        def __generator(a, s, m, x):
            return (a/s) * pow((x-m)/s, -1-a)*_exp(-pow((x-m)/s, -a))

        if plot:
            x = np.linspace(-interval, interval, int(threshold))
            y = np.array(
                [__generator(self.shape, self.scale, self.location, i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return __generator(self.shape, self.scale, self.location, self.randvar)

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
            either cumulative distribution evaluation for some point or plot of Fréchet distribution.
        """
        def __generator(a, s, m, x): return _exp(-pow((x-m)/s, -a))

        if plot:
            x = np.linspace(-interval, interval, int(threshold))
            y = np.array(
                [__generator(self.shape, self.scale, self.location, i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return __generator(self.shape, self.scale, self.location, self.randvar)

    def pvalue(self, x_lower=0, x_upper=None) -> Optional[float]:
        """
        Args:

            x_lower(float): defaults to 0. Defines the lower value of the distribution. Optional.
            x_upper(float): defaults to None. If not defined defaults to random variable x. Optional.

            Note: definition of x_lower and x_upper are only relevant when probability is between two random variables.
            Otherwise, the default random variable is x.

        Returns:
            p-value of the Logit distribution evaluated at some random variable.
        """
        if x_lower < 0:
            raise ValueError(
                'x_lower should be a positive number. X_lower:{}'.format(x_lower))
        if x_upper == None:
            x_upper = self.randvar
        if x_lower > x_upper:
            raise ValueError(
                'lower bound should be less than upper bound. Entered values: x_lower:{} x_upper:{}'.format(x_lower, x_upper))

        def __cdf(a, s, m, x): return _exp(-pow((x-m)/s, -a))
        return __cdf(self.shape, self.scale, self.location, x_upper)-__cdf(self.shape, self.scale, self.location, x_lower)

    def mean(self) -> float:
        """
        Returns: Mean of the Fréchet distribution.
        """
        if self.shape > 1:
            return self.location + (self.scale*_gamma(1 - 1/self.shape))
        return np.inf

    def median(self) -> float:
        """
        Returns: Median of the Fréchet distribution.
        """
        return self.location + (self.scale/pow(_log(2), 1/self.shape))

    def mode(self) -> float:
        """
        Returns: Mode of the Fréchet distribution.
        """
        return self.location + self.scale*(self.shape/pow(1 + self.shape, 1/self.shape))

    def var(self) -> Union[float, str]:
        """
        Returns: Variance of the Fréchet distribution.
        """
        a = self.shape
        s = self.scale
        if a > 2:
            return pow(s, 2)*(_gamma(1-2/a)-pow(_gamma(1-1/a), 2))
        return "infinity"

    def std(self) -> Union[float, str]:
        """
        Returns: Standard devtiation of the Fréchet distribution.
        """
        if self.var() == "infinity":
            return "infinity"
        return _sqrt(self.var())

    def skewness(self) -> Union[float, str]:
        """
        Returns: Skewness of the Fréchet distribution.
        """
        a = self.shape
        if a > 3:
            return (_gamma(1-3/a)-3*_gamma(1-2/a)*_gamma(1-1/a)+2*_gamma(1-1/a)**3)/pow(_gamma(1-2/a)-pow(_gamma(1-1/a), 2), 3/2)
        return "infinity"

    def kurtosis(self) -> Union[float, str]:
        """
        Returns: Kurtosis of the Fréchet distribution.
        """
        a = self.shape
        if a > 4:
            return -6+(_gamma(1-4/a)-4*_gamma(1-3/a)*_gamma(1-1/a)+3*pow(_gamma(1-2/a), 2))/pow(_gamma(1-2/a)-pow(_gamma(1-1/a), 2), 2)
        return "infinity"

    def summary(self, display=False) -> Union[None, Tuple[str, str, str, str, str, str, str]]:
        """
        Returns:  summary statistic regarding the Weilbul Inverse distribution which contains the following parts of the distribution:
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
        Summary statistic regarding the Weilbul Inverse distribution which contains the following parts of the distribution:
        (mean, median, mode, var, std, skewness, kurtosis).

        Returns:
            Dict[str, Union[float, int, str]]: [description]
        """
        return {
            'mean': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.var(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }

