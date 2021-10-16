try:
    from scipy.special import gamma as _gamma, gammainc as _gammainc, digamma as _digamma
    import numpy as np
    from typing import Union, Tuple, Dict
    from math import sqrt as _sqrt, log as _log
    from _base import SemiInfinite
except Exception as e:
    print(f"some modules are missing {e}")


class Erlang(SemiInfinite):
    """
    This class contains methods concerning Erlang Distirbution.
    Args:

        shape(int | x>0): shape
        rate(float | x>=0): rate
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
        - keys for returning a dictionary of summary statistics.

    Reference:
    - Wikipedia contributors. (2021, January 6). Erlang distribution. In Wikipedia, The Free Encyclopedia.
    Retrieved 09:38, January 8, 2021, from https://en.wikipedia.org/w/index.php?title=Erlang_distribution&oldid=998655107
    - Weisstein, Eric W. "Erlang Distribution." From MathWorld--A Wolfram Web Resource.
    https://mathworld.wolfram.com/ErlangDistribution.html
    """

    def __init__(self, shape: int, rate: float, randvar: float):
        if randvar < 0:
            raise ValueError(
                f'random variable should only be in between 0 and 1. Entered value: {randvar}')
        if isinstance(shape, int) == False and shape > 0:
            raise TypeError(
                'shape parameter should be an integer greater than 0.')
        if rate < 0:
            raise ValueError(
                f'beta parameter(rate) should be a positive number. Entered value: {rate}')

        self.shape = shape
        self.rate = rate
        self.randvar = randvar

    def pdf(self,
            plot=False,
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
            either probability density evaluation for some point or plot of Erlang distribution.
        """
        def __generator(shape, rate, x): return (
            pow(rate, shape)*pow(x, shape-1)*np.exp(-rate*x))/np.math.factorial((shape-1))

        if plot:
            x = np.linspace(0, 1, int(threshold))
            y = np.array([__generator(self.shape, self.rate, i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return __generator(self.shape, self.rate, self.randvar)

    def cdf(self,
            plot=False,
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
            either cumulative distribution evaluation for some point or plot of Erlang distribution.
        """
        def __generator(shape, rate, x): return _gammainc(
            shape, rate*x)/np.math.factorial(shape-1)
        if plot:
            x = np.linspace(0, 1, int(threshold))
            y = np.array([__generator(self.shape, self.rate, i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)

        return __generator(self.shape, self.rate, self.randvar)

    def pvalue(self, x_lower=0, x_upper=None) -> Optional[float]:
        """
        Args:

            x_lower(float): defaults to 0. Defines the lower value of the distribution. Optional.
            x_upper(float): defaults to None. If not defined defaults to random variable x. Optional.

            Note: definition of x_lower and x_upper are only relevant when probability is between two random variables.
            Otherwise, the default random variable is x.

        Returns:
            p-value of the Erlang distribution evaluated at some random variable.
        """
        if x_upper == None:
            x_upper = self.randvar
        if x_lower > x_upper:
            raise Exception(
                'lower bound should be less than upper bound. Entered values: x_lower:{} x_upper:{}'.format(x_lower, x_upper))

        def __cdf(shape, rate, x): return _gammainc(
            shape, rate*x)/np.math.factorial(shape-1)

        return __cdf(self.shape, self.rate, x_upper)-__cdf(self.shape, self.rate, x_lower)

    def mean(self) -> float:
        """
        Returns: Mean of the Erlang distribution.
        """
        return self.shape/self.rate

    def median(self) -> str:
        """
        Returns: Median of the Erlang distribution.
        """
        return "no simple closed form"

    def mode(self) -> Union[float, str]:
        """
        Returns: Mode of the Erlang distribution.
        """
        return (1/self.rate)*(self.shape-1)

    def var(self) -> float:
        """
        Returns: Variance of the Erlang distribution.
        """
        return self.shape/pow(self.rate, 2)

    def std(self) -> float:
        """
        Returns: Standard deviation of the Eerlang distribution.
        """
        return _sqrt(self.var())

    def skewness(self) -> float:
        """
        Returns: Skewness of the Erlang distribution.
        """
        return 2/_sqrt(self.shape)

    def kurtosis(self) -> float:
        """
        Returns: Kurtosis of the Erlang distribution.
        """
        return 6/self.shape

    def entropy(self) -> float:
        """
        Returns: differential entropy of the Erlang distribution.

        Reference: Park, S.Y. & Bera, A.K.(2009). Maximum entropy autoregressive conditional heteroskedasticity model. Elsivier.
        link: http://wise.xmu.edu.cn/uploadfiles/paper-masterdownload/2009519932327055475115776.pdf
        """
        k = self.shape
        _lambda = self.rate
        return (1-k)*_digamma(k)+_log(_gamma(k)/_lambda)+k

    def summary(self, display=False) -> Union[None, Tuple[str, str, str, str, str, str, str]]:
        """
        Returns:  summary statistic regarding the Erlang distribution which contains the following parts of the distribution:
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
        Summary statistic regarding the Erlang distribution which contains the following parts of the distribution:
        (mean, median, mode, var, std, skewness, kurtosis).

        Returns:
            Dict[str, Union[float, int, str]]: [description]
        """
        return {
            'mean': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.var(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }

