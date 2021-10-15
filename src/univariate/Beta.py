try:
    from scipy.special import beta as _beta, betainc as _betainc, digamma as _digamma
    import numpy as np
    from typing import Union, Tuple, Dict
    from math import sqrt as _sqrt, log as _log
    from _base import Base
except Exception as e:
    print(f"some modules are missing {e}")


class Beta(Base):
    """
    This class contains methods concerning Beta Distirbution.
    Args:

        alpha(float | x>0): shape
        beta(float | x>0): shape
        randvar(float | [0,1]): random variable

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
    - Wikipedia contributors. (2021, January 8). Beta distribution. In Wikipedia, The Free Encyclopedia.
    Retrieved 07:21, January 8, 2021, from https://en.wikipedia.org/w/index.php?title=Beta_distribution&oldid=999043368
    """

    def __init__(self, alpha: float, beta: float, randvar: float):
        if randvar < 0 | randvar > 1:
            raise ValueError(
                f'random variable should only be in between 0 and 1. Entered value: {randvar}')
        if alpha < 0:
            raise ValueError(
                f'alpha parameter(shape) should be a positive number. Entered value:{alpha}')
        if beta < 0:
            raise ValueError(
                f'beta parameter(shape) should be a positive number. Entered value:{beta}')

        self.alpha = alpha
        self.beta = beta
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
            either probability density evaluation for some point or plot of Beta distribution.
        """
        def __generator(a, b, x): return (
            pow(x, a-1)*pow(1-x, b-1))/_beta(a, b)

        if plot:
            x = np.linspace(0, 1, int(threshold))
            y = np.array([__generator(self.alpha, self.beta, i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return __generator(self.alpha, self.beta, self.randvar)

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
            either cumulative distribution evaluation for some point or plot of Beta distribution.
        """
        def __generator(a, b, x): return _betainc(a, b, x)
        if plot:
            x = np.linspace(0, 1, int(threshold))
            y = np.array([__generator(self.a, self.b, self.c, i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return __generator(self.a, self.b, self.c, self.randvar)

    def pvalue(self, x_lower=0, x_upper=None) -> Optional[float]:
        """
        Args:

            x_lower(float): defaults to 0. Defines the lower value of the distribution. Optional.
            x_upper(float): defaults to None. If not defined defaults to random variable x. Optional.

            Note: definition of x_lower and x_upper are only relevant when probability is between two random variables.
            Otherwise, the default random variable is x.

        Returns:
            p-value of the Beta distribution evaluated at some random variable.
        """
        if x_upper == None:
            x_upper = self.randvar
        if x_lower > x_upper:
            raise ValueError(
                'lower bound should be less than upper bound. Entered values: x_lower:{} x_upper:{}'.format(x_lower, x_upper))

        def __cdf(a, b, x): return _betainc(a, b, x)
        return __cdf(self.alpha, self.beta, x_upper)-__cdf(self.alpha, self.beta, x_lower)

    def mean(self) -> str:
        """
        Returns: Mean of the Beta distribution.
        """
        return "currently unsupported."

    def median(self) -> float:
        """
        Returns: Median of the Beta distribution.
        """
        # warning: not yet validated.
        return _betainc(self.alpha, self.beta, 0.5)

    def mode(self) -> str:
        """
        Returns: Mode of the Beta distribution.
        """
        return "currently unsupported"

    def var(self) -> str:
        """
        Returns: Variance of the Beta distribution.
        """
        return "currently unsupported"

    def skewness(self) -> float:
        """
        Returns: Skewness of the Beta distribution.
        """
        alpha = self.alpha
        beta = self.beta
        return (2*(beta-alpha)*_sqrt(alpha+beta+1))/((alpha+beta+2)*_sqrt(alpha*beta))

    def kurtosis(self) -> float:
        """
        Returns: Kurtosis of the Beta distribution.
        """
        alpha = self.alpha
        beta = self.beta
        temp_up = 6*((alpha-beta)**2*(alpha+beta+1)-alpha*beta*(alpha+beta+2))
        return temp_up/(alpha*beta*(alpha+beta+2)*(alpha+beta+3))

    def entropy(self) -> float:
        """
        Returns: differential entropy of the Beta distribution.

        Reference: Park, S.Y. & Bera, A.K.(2009). Maximum entropy autoregressive conditional heteroskedasticity model. Elsivier.
        link: http://wise.xmu.edu.cn/uploadfiles/paper-masterdownload/2009519932327055475115776.pdf
        """
        alpha = self.alpha
        beta = self.beta
        return _log(_beta(alpha, beta))-(alpha-1)*(_digamma(alpha)-_digamma(alpha+beta))-(beta-1)*(_digamma(beta)-_digamma(alpha+beta))

    def summary(self, display=False) -> Union[None, Tuple[str, str, str, str, str, str, str]]:
        """
        Returns:  summary statistic regarding the Beta distribution which contains the following parts of the distribution:
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
        Summary statistic regarding the Beta distribution which contains the following parts of the distribution:
        (mean, median, mode, var, std, skewness, kurtosis).

        Returns:
            Dict[str, Union[float, Tuple[float]]]: [description]
        """
        return {
            'mean': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.var(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }