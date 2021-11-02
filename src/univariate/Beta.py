try:
    from scipy.special import beta as _beta, betainc as _betainc, digamma as _digamma
    import numpy as _np
    from typing import Union, Tuple, Dict, List
    from math import sqrt as _sqrt, log as _log
    from univariate._base import BoundedInterval
except Exception as e:
    print(f"some modules are missing {e}")


class Beta(BoundedInterval):
    """
    This class contains methods concerning Beta Distirbution.
    Args:

        alpha(float | x>0): shape
        beta(float | x>0): shape
        randvar(float | [0,1]): random variable

    Reference:
    - Wikipedia contributors. (2021, January 8). Beta distribution. In Wikipedia, The Free Encyclopedia.
    Retrieved 07:21, January 8, 2021, from https://en.wikipedia.org/w/index.php?title=Beta_distribution&oldid=999043368
    """

    def __init__(self, alpha: float, beta: float, randvar: float):
        if randvar < 0 or randvar > 1:
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

    def pdf(self, x: Union[List[float], _np.ndarray] = None) -> Union[float, _np.ndarray]:
        """
        Args:

            x (List[float], numpy.ndarray): random variable or list of random variables

        Returns:
            either probability density evaluation for some point or plot of Beta distribution.
        """
        a = self.a
        b = self.b
        c = self.c
        randvar = self.randvar

        if x is not None:
            if not isinstance(x, (_np.ndarray, List)):
                raise TypeError(f'parameter x only accepts List types or numpy.ndarray')
            else:
                x = _np.array(x)
                return (_np.power(x, a-1)*_np.power(1-x, b-1))/_beta(a, b)

        return (pow(randvar, a-1)*pow(1-randvar, b-1))/_beta(a, b)

    def cdf(self, x: Union[List[float], _np.ndarray] = None) -> Union[float, _np.ndarray]:
        """
        Args:

            x (List[float], numpy.ndarray): random variable or list of random variables

        Returns:
            either cumulative distribution evaluation for some point or plot of Beta distribution.
        """
        a = self.a
        b = self.b
        c = self.c
        randvar = self.randvar

        if x is not None:
            if not isinstance(x, (_np.ndarray, List)):
                raise TypeError(f'parameter x only accepts List types or numpy.ndarray')
            else:
                return _betainc(a, b, x)

        return _betainc(a, b, x)

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
    
    def std(self) -> str:
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

    def summary(self) -> Dict[str, Union[float, str]]:
        """
        Summary statistic regarding the Beta distribution which contains the following parts of the distribution:
        (mean, median, mode, var, std, skewness, kurtosis).

        Returns:
            Dict[str, Union[float, str]]
        """
        return {
            'mean': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.var(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }
