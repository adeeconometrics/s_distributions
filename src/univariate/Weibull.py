try:
    from scipy.special import gamma as _gamma
    from numpy import euler_gamma as _euler_gamma
    import numpy as np
    from math import sqrt as _sqrt, log as _log
    from typing import Union, Tuple, Dict
    from _base import SemiInfinite
except Exception as e:
    print(f"some modules are missing {e}")


class Weibull(SemiInfinite):
    """
    This class contains methods concerning Weibull Distirbution. Also known as Fréchet distribution.
    Args:

        shape(float | [0, infty)): mean parameter
        scale(float | [0, infty)): standard deviation
        randvar(float | [0, infty)): random variable. Optional. Use when cdf and pdf or p value of interest is desired.

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
    - Wikipedia contributors. (2020, December 13). Weibull distribution. In Wikipedia, The Free Encyclopedia.
    Retrieved 11:32, December 28, 2020, from https://en.wikipedia.org/w/index.php?title=Weibull_distribution&oldid=993879185
    """

    def __init__(self, shape: Union[float, int], scale: Union[float, int], randvar: Union[float, int] = 0.5):
        if shape < 0 or scale < 0 or randvar < 0:
            raise ValueError(
                f'all parameters should be a positive number. Entered values: shape: {shape}, scale{scale}, randvar{randvar}')
        self.scale = scale
        self.shape = shape
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
            either probability density evaluation for some point or plot of Weibull distribution.
        """
        def __generator(_lambda, k, x):
            if x < 0:
                return 0
            if x >= 0:
                return pow((k/_lambda)*(x/_lambda), k-1)*np.exp(-pow(x/_lambda, k))
        if plot:
            x = np.linspace(-interval, interval, int(threshold))
            y = np.array([__generator(self.scale, self.shape, i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return __generator(self.scale, self.shape, self.randvar)

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
            either cumulative distribution evaluation for some point or plot of Weibull distribution.
        """
        def __generator(_lambda, k, x):
            if x < 0:
                return 0
            if x >= 0:
                return 1-np.exp(-pow(x/_lambda, k))

        if plot:
            x = np.linspace(-interval, interval, int(threshold))
            y = np.array([__generator(self.scale, self.shape, i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return __generator(self.scale, self.shape, self.randvar)

    def pvalue(self, x_lower=0, x_upper=None) -> Optional[float]:
        """
        Args:

            x_lower(float): defaults to 0. Defines the lower value of the distribution. Optional.
            x_upper(float): defaults to None. If not defined defaults to random variable x. Optional.

            Note: definition of x_lower and x_upper are only relevant when probability is between two random variables.
            Otherwise, the default random variable is x.

        Returns:
            p-value of the Weilbull distribution evaluated at some random variable.
        """
        if x_lower < 0:
            raise ValueError(
                f'x_lower should be a positive number. X_lower:{x_lower}')
        if x_upper == None:
            x_upper = self.randvar
        if x_lower > x_upper:
            raise ValueError(
                f'lower bound should be less than upper bound. Entered values: x_lower: {x_lower}, x_upper:{x_upper}')

        def __cdf(_lambda, k, x):
            if x < 0:
                return 0
            if x >= 0:
                return 1-np.exp(-pow(x/_lambda, k))

        return __cdf(self.location, self.shape, x_upper)-__cdf(self.location, self.shape, x_lower)

    def mean(self) -> float:
        """
        Returns: Mean of the Weibull distribution.
        """
        return self.scale*_gamma(1+(1/self.shape))

    def median(self) -> float:
        """
        Returns: Median of the Weibull distribution.
        """
        return self.scale*pow(_log(2), 1/self.shape)

    def mode(self) -> Union[float, int]:
        """
        Returns: Mode of the Weibull distribution.
        """
        if self.shape > 1:
            return self.scale*pow((self.shape-1)/self.shape, 1/self.shape)
        return 0

    def var(self) -> float:
        """
        Returns: Variance of the Weibull distribution.
        """
        return pow(self.scale, 2) * pow(_gamma(1+2/self.shape) - _gamma(1+1/self.shape), 2)

    def std(self) -> float:
        """
        Returns: Standard deviation of the Weilbull distribution
        """
        return _sqrt(pow(self.scale, 2) * pow(_gamma(1+2/self.shape) - _gamma(1+1/self.shape), 2))

    def entropy(self) -> float:
        """
        Returns: differential entropy of the Weilbull distribution.

        Reference: Park, S.Y. & Bera, A.K.(2009). Maximum entropy autoregressive conditional heteroskedasticity model. Elsivier.
        link: http://wise.xmu.edu.cn/uploadfiles/paper-masterdownload/2009519932327055475115776.pdf
        """
        return (self.scale+1) * _euler_gamma/self.scale + _log(self.shape/self.scale) + 1

    def summary(self, display=False) -> Union[None, Tuple[str, str, str, str, str, str, str]]:
        """
        Returns:  summary statistic regarding the ChiSquare-distribution which contains the following parts of the distribution:
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
        Summary statistic regarding the ChiSquare-distribution which contains the following parts of the distribution:
        (mean, median, mode, var, std, skewness, kurtosis).

        Returns:
            Dict[str, Union[float, int, str]]: [description]
        """
        return {
            'mean': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.var(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }

