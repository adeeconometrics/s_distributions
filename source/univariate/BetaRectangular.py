
try:
    from scipy.special import betainc as _betainc, gamma as _gamma
    import numpy as np
    from typing import Union, Tuple, Dict
    from math import sqrt as _sqrt, log as _log
    from univariate._base import BoundedInterval
except Exception as e:
    print(f"some modules are missing {e}")


class BetaRectangular(BoundedInterval):
    """
    This class contains methods concerning Beta-rectangular Distirbution.
    Thus it is a bounded distribution that allows for outliers to have a greater chance of occurring than does the beta distribution.

    .. math::
        \\text{BetaRectangulat}(x,\\alpha ,\\beta ,\\theta )={\\begin{cases}{\\frac{\\theta \\Gamma (\\alpha +\\beta )}{\\Gamma (\\alpha )\\Gamma (\\beta )}}{\\frac{(x-a)^{{\\alpha -1}}(b-x)^{{\\beta -1}}}{(b-a)^{{\\alpha +\\beta +1}}}}+{\\frac{1-\\theta }{b-a}}&{\mathrm{for}}\ a\leq x\leq b,\\\[8pt]0&{\\mathrm{for}}\ x<a\{\\mathrm{or}}\ x>b\\end{cases}}
    Args:

        alpha(float): shape parameter
        beta (float): shape parameter
        theta(float | 0<x<1): mixture parameter
        min(float): lower bound
        max(float): upper bound
        randvar(float | alpha<=x<=beta): random variable

    Reference:
        .. [#] Wikipedia contributors. (2020, December 7). Beta rectangular distribution. https://en.wikipedia.org/w/index.php?title=Beta_rectangular_distribution&oldid=992814814
    """

    def __init__(self, alpha: float, beta: float, theta: float, min: float, max: float, randvar: float):
        if alpha < 0 or beta < 0:
            raise ValueError(
                'alpha and beta parameter should not be less that 0. Entered values: alpha: {alpha}, beta: {beta}}')
        if theta < 0 or theta > 1:
            raise ValueError(
                'random variable should only be in between 0 and 1. Entered value: {theta}')
        if randvar < min and randvar > max:  # should only return warning
            raise ValueError(
                f'random variable should be between alpha and beta shape parameters. Entered value:{randvar}')

        self.alpha = alpha
        self.beta = beta
        self.theta = theta
        self.min = min
        self.max = max
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
            either probability density evaluation for some point or plot of Beta-rectangular distribution.
        """
        def __generator(a, b, alpha, beta, theta, x):
            if x > a or x < b:
                return (theta*_gamma(alpha+beta)/(_gamma(alpha)*_gamma(beta))*(pow(x-a, alpha-1)*pow(b-x, beta-1))/(pow(b-a, alpha+beta+1)))+(1-theta)/(b-a)
            return 0

        if plot:
            x = np.linspace(0, 1, int(threshold))
            y = np.array([__generator(self.min, self.max, self.alpha,
                                      self.beta, self.theta, i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return __generator(self.min, self.max, self.alpha, self.beta, self.theta, self.randvar)

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
            either cumulative distribution evaluation for some point or plot of Beta-rectangular distribution.
        """
        def __generator(a, b, alpha, beta, theta, x):
            if x <= a:
                return 0
            elif x > a | x < b:
                z = (b)/(b-a)
                return theta*_betainc(alpha, beta, z)+((1-theta)*(x-a))/(b-a)
            else:
                return 1

        if plot:
            if interval < 0:
                raise ValueError(
                    'interval parameter should be a positive number. Entered Value {}'.format(interval))
            x = np.linspace(0, interval, int(threshold))
            y = np.array([__generator(self.min, self.max, self.alpha,
                                      self.beta, self.theta, i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return __generator(self.min, self.max, self.alpha, self.beta, self.theta, self.randvar)

    def mean(self) -> float:
        """
        Returns: Mean of the Beta-rectangular distribution.
        """
        alpha = self.alpha
        beta = self.beta
        theta = self.theta
        a = self.min
        b = self.max
        return a+(b-a)*((theta*alpha)/(alpha+beta)+(1-theta)/2)

    def var(self) -> float:
        """
        Returns: Variance of the Beta-rectangular distribution.
        """
        alpha = self.alpha
        beta = self.beta
        theta = self.theta
        a = self.min
        b = self.max
        k = alpha+beta
        return (b-a)**2*((theta*alpha*(alpha+1))/(k*(k+1))+(1-theta)/3-(k+theta*(alpha-beta))**2/(4*k**2))

    def std(self) -> float:
        """
        Returns: Standard deviation of the Beta-rectangular distribution.
        """
        return _sqrt(self.var())

    def summary(self) -> Dict[str, Union[float, Tuple[float]]]:
        """
        Summary statistic regarding the Beta-rectangular distribution which contains the following parts of the distribution:
        (mean, median, mode, var, std, skewness, kurtosis).
        """
        return {
            'mean': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.var(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }
