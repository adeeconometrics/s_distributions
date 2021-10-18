try:
    from scipy.special import logit as _logit, erf as _erf
    import numpy as _np
    from math import sqrt as _sqrt, pi as _pi, exp as _exp
    from typing import Union, Tuple, Dict, List
    from _base import BoundedInterval
except Exception as e:
    print(f"some modules are missing {e}")


class LogitNormal(BoundedInterval):
    """
    This class contains methods concerning Logit Normal Distirbution.
    Args:

        sq_scale (float): squared scale parameter
        location(float): location parameter
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
        - keys for returning a dictionary of summary statistics.

    Reference:
    - Wikipedia contributors. (2020, December 9). Logit-normal distribution. In Wikipedia, The Free Encyclopedia.
    Retrieved 07:44, December 30, 2020, from https://en.wikipedia.org/w/index.php?title=Logit-normal_distribution&oldid=993237113
    """

    def __init__(self, sq_scale: Union[float, int], location: Union[float, int], randvar: Union[float, int]):
        if randvar < 0 or randvar > 1:
            raise ValueError(
                f'random variable should only be in between (0,1). Entered value: randvar:{randvar}')
        self.sq_scale = sq_scale
        self.location = location
        self.randvar = randvar

    def pdf(self, x: Union[List[float], _np.ndarray] = None) -> Union[float, _np.ndarray]:
        """
        Args:

            x (List[float], numpy.ndarray): random variable or list of random variables

        Returns:
            either probability density evaluation for some point or plot of Logit Normal distribution.
        """
        mu = self.location
        sig = self.sq_scale 
        randvar = self.randvar

        if x is not None:
            if not (isinstance(x, _np.ndarray)) and issubclass(x, List):
                raise TypeError(f'parameter x only accepts List types or numpy.ndarray')
            else:
                x = _np.array(x)
                return (1/(sig*_sqrt(2*_pi)))* _np.exp(-(_np.power(_logit(x)-mu, 2)/(2*pow(sig, 2)))) * 1/(x*(1-x))
        
        return (1/(sig*_sqrt(2*_pi)))* _exp(-pow(_logit(x)-mu, 2)/(2*pow(sig, 2))) * 1/(x*(1-x))

    def cdf(self, x: Union[List[float], _np.ndarray] = None) -> Union[float, _np.ndarray]:
        """
        Args:

            x (List[float], numpy.ndarray): random variable or list of random variables

        Returns:
            either cumulative distribution evaluation for some point or plot of Logit Normal distribution.
        """
        mu = self.location
        sig = self.sq_scale 
        randvar = self.randvar

        def __generator(mu:float, sig:float, x:Union[float, _np.ndarray]) -> Union[float, _np.ndarray]:
            return 1/2 * (1+_erf((_logit(x)-mu)/_sqrt(2*pow(sig, 2))))

        if x is not None:
            if not (isinstance(x, _np.ndarray)) and issubclass(x, List):
                raise TypeError(f'parameter x only accepts List types or numpy.ndarray')
            else:
                x = _np.array(x)
                return __generator(mu, sig, x)

        return __generator(mu, sig, randvar)

    def pvalue(self, x_lower=0, x_upper=None) -> Union[float, int, None]:
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
                f'x_lower should be a positive number. X_lower:{x_lower}')
        if x_upper == None:
            x_upper=self.randvar
        if x_lower > x_upper:
            raise ValueError(
                f'lower bound should be less than upper bound. Entered values: x_lower:{x_lower} x_upper:{x_upper}')
        def __cdf(mu, sig, x):
            return 1/2 * (1+_erf((_logit(x)-mu)/(_sqrt(2*pow(sig, 2)))))
        return __cdf(self.location, self.sq_scale, x_upper)-__cdf(self.location, self.sq_scale, x_lower)

    def mean(self) -> str:
        """
        Returns: Mean of the Logit Normal distribution.
        """
        return "no analytical solution"

    def mode(self) -> str:
        """
        Returns: Mode of the Logit Normal distribution.
        """
        return "no analytical solution"

    def var(self) -> str:
        """
        Returns: Variance of the Logit Normal distribution.
        """
        return "no analytical solution"

    def std(self) -> str:
        """
        Returns: Standard deviation of the Logit Normal distribution.
        """
        return "no analytical solution"

    def entropy(self) -> str:
        """
        Returns: differential entropy of Logit Normal distribution.

        Reference: Park, S.Y. & Bera, A.K.(2009). Maximum entropy autoregressive conditional heteroskedasticity model. Elsivier.
        link: http://wise.xmu.edu.cn/uploadfiles/paper-masterdownload/2009519932327055475115776.pdf
        """
        return "unsupported"

    def summary(self, display=False) -> Union[None, Tuple[str, str, str, str, str, str, str]]:
        """
        Returns:  summary statistic regarding the LogitNormal distribution which contains the following parts of the distribution:
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

    def keys(self) -> Dict[str, str]:
        """
        Summary statistic regarding the LogitNormal distribution which contains the following parts of the distribution:
        (mean, median, mode, var, std, skewness, kurtosis).

        Returns:
            Dict[str, str]: [description]
        """
        return {
            'mean': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.var(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }
