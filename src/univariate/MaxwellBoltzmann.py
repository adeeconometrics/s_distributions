try:
    from scipy.special import erf as _erf
    from numpy import euler_gamma as _euler_gamma
    import numpy as _np
    from typing import Union, Tuple, Dict, List
    from math import sqrt as _sqrt, log as _log, pi as _pi, exp as _exp
    from univariate._base import SemiInfinite
except Exception as e:
    print(f"some modules are missing {e}")


class Maxwell_Boltzmann(SemiInfinite):
    """
    This class contains methods concerning Maxwell-Boltzmann Distirbution.
    Args:

        a(int | x>0): parameter
        randvar(float | x>=0): random variable. Optional. Use when cdf and pdf or p value of interest is desired.

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
    - Wikipedia contributors. (2021, January 12). Maxwell–Boltzmann distribution. In Wikipedia, The Free Encyclopedia.
    Retrieved 01:02, January 14, 2021, from https://en.wikipedia.org/w/index.php?title=Maxwell%E2%80%93Boltzmann_distribution&oldid=999883013
    """

    def __init__(self, a: int, randvar=0.5):
        if randvar < 0:
            raise ValueError(
                'random variable should be a positive number. Entered value: {}'.format(randvar))
        if a < 0:
            raise ValueError(
                'parameter a should be a positive number. Entered value:{}'.format(a))
        if isinstance(a, int) == False:
            raise TypeError('parameter should be in type int')

        self.a = a
        self.randvar = randvar

    def pdf(self, x: Union[List[float], _np.ndarray] = None) -> Union[float, _np.ndarray]:
        """
        Args:

            x (List[float], numpy.ndarray): random variable or list of random variables

        Returns:
            either probability density evaluation for some point or plot of Maxwell-Boltzmann distribution.
        """ 
        a = self.a
        randvar = self.randvar

        def __generator(a, x): return _sqrt(2/_pi)*(x**2*_np.exp(-x**2/(2*a**2)))/(a**3)

        if x is not None:
            if not (isinstance(x, _np.ndarray)) and issubclass(x, List):
                raise TypeError(f'parameter x only accepts List types or numpy.ndarray')
            else:
                x = _np.array(x)
                return _sqrt(2/_pi)*(x**2*_np.exp(-x**2/(2*a**2)))/a**3

        return _sqrt(2/_pi)*(randvar**2*_exp(-randvar**2/(2*a**2)))/a**3

    def cdf(self, x: Union[List[float], _np.ndarray] = None) -> Union[float, _np.ndarray]:
        """
        Args:

            x (List[float], numpy.ndarray): random variable or list of random variables

        Returns:
            either cumulative distirbution evaluation for some point or plot of Maxwell-Boltzmann distribution.
        """ 
        a = self.a
        randvar = self.randvar

        if x is not None:
            if not (isinstance(x, _np.ndarray)) and issubclass(x, List):
                raise TypeError(f'parameter x only accepts List types or numpy.ndarray')
            else:
                x = _np.array(x)
                _x = _np.power(x,2)
                return _erf(x/(_sqrt(2)*a))-_sqrt(2/_pi)*(_x*_np.exp(-_x/(2*a**2)))/(a)

        return _erf(x/(_sqrt(2)*a))- _sqrt(2/_pi)*(randvar**2*_exp(-randvar**2/(2*a**2)))/(a)

    def pvalue(self, x_lower=0, x_upper=None) -> Optional[float]:
        """
        Args:

            x_lower(float): defaults to 0. Defines the lower value of the distribution. Optional.
            x_upper(float): defaults to None. If not defined defaults to random variable x. Optional.

            Note: definition of x_lower and x_upper are only relevant when probability is between two random variables.
            Otherwise, the default random variable is x.

        Returns:
            p-value of the Maxwell-Boltzmann distribution evaluated at some random variable.
        """
        if x_upper == None:
            x_upper = self.randvar
        if x_lower > x_upper:
            raise Exception(
                'lower bound should be less than upper bound. Entered values: x_lower:{} x_upper:{}'.format(x_lower, x_upper))

        def __cdf(a:float, x:float) -> float: 
            return _erf(x/(_sqrt(2)*a))-_sqrt(2/_pi)*(x**2*_exp(-x**2/(2*a**2)))/(a)
        return __cdf(self.a, x_upper)-__cdf(self.a, x_lower)

    def mean(self) -> float:
        """
        Returns: Mean of the Maxwell-Boltzmann distribution.
        """
        return 2*self.a*_sqrt(2/_pi)

    def median(self) -> Union[float, str]:
        """
        Returns: Median of the Maxwell-Boltzmann distribution.
        """
        return "currently unsupported"

    def mode(self) -> float:
        """
        Returns: Mode of the Maxwell-Boltzmann distribution.
        """
        return _sqrt(2)*self.a

    def var(self) -> float:
        """
        Returns: Variance of the Maxwell-Boltzmann distribution.
        """
        return (self.a**2*(3*_pi-8))/_pi

    def std(self) -> float:
        """
        Returns: Standard deviation of the Maxwell-Boltzmann distribution
        """
        return _sqrt((self.a**2*(3*_pi-8))/_pi)

    def skewness(self) -> float:
        """
        Returns: Skewness of the Maxwell-Boltzmann distribution.
        """
        return (2*_sqrt(2)*(16-5*_pi))/_np.power((3*_pi-8), 3/2)

    def kurtosis(self) -> float:
        """
        Returns: Kurtosis of the Maxwell-Boltzmann distribution.
        """
        return 4*((-96+40*_pi-3*_pi**2)/(3*_pi-8)**2)

    def entropy(self) -> float:
        """
        Returns: differential entropy of the Maxwell-Boltzmann distribution.

        Reference: Park, S.Y. & Bera, A.K.(2009). Maximum entropy autoregressive conditional heteroskedasticity model. Elsivier.
        link: http://wise.xmu.edu.cn/uploadfiles/paper-masterdownload/2009519932327055475115776.pdf
        """
        a = self.a
        return _log(a*_sqrt(2*_pi)+_euler_gamma-0.5)

    def summary(self, display=False) -> Union[None, Tuple[str, str, str, str, str, str, str]]:
        """
        Returns:  summary statistic regarding the Maxwell-Boltzmanndistribution which contains the following parts of the distribution:
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
        Summary statistic regarding the Maxwell-Boltzmanndistribution which contains the following parts of the distribution:
        (mean, median, mode, var, std, skewness, kurtosis).

        Returns:
            Dict[str, Union[float, Tuple[float]]]: [description]
        """
        return {
            'mean': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.var(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }

