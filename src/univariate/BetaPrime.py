try:
    from scipy.special import beta as _beta, betainc as _betainc
    import numpy as _np
    from typing import Union, Tuple, Dict, List
    from math import sqrt as _sqrt, log as _log
    from univariate._base import SemiInfinite
except Exception as e:
    print(f"some modules are missing {e}")


class BetaPrime(SemiInfinite):
    """
    This class contains methods concerning Beta prime Distirbution.
    Args:

        alpha(float | x>0): shape
        beta(float | x>0): shape
        randvar(float | x>=0): random variable

    Reference:
    - Wikipedia contributors. (2020, October 8). Beta prime distribution. In Wikipedia, The Free Encyclopedia.
    Retrieved 09:40, January 8, 2021, from https://en.wikipedia.org/w/index.php?title=Beta_prime_distribution&oldid=982458594
    """

    def __init__(self, alpha: float, beta: float, randvar: float):
        if randvar < 0:
            raise ValueError(
                f'random variable should not be less then 0. Entered value: {randvar}')
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
            either probability density evaluation for some point or plot of Beta prime distribution.
        """
        a = self.alpha
        b = self.beta
        randvar = self.randvar

        if x is not None:
            if not isinstance(x, (_np.ndarray, List)):
                raise TypeError(f'parameter x only accepts List types or numpy.ndarray')
            else:
                x = _np.array(x)
                return _np.power(x, a-1)*_np.power(1+x, -a-b)/_beta(a, b)

        return pow(randvar, a-1)*pow(1+randvar, -a-b)/_beta(a, b)

    def cdf(self, x: Union[List[float], _np.ndarray] = None) -> Union[float, _np.ndarray]:
        """
        Args:

            x (List[float], numpy.ndarray): random variable or list of random variables

        Returns:
            either cumulative distribution evaluation for some point or plot of Beta prime distribution.
        """
        a = self.alpha
        b = self.beta
        randvar = self.randvar

        if x is not None:
            if not isinstance(x, (_np.ndarray, List)):
                raise TypeError(f'parameter x only accepts List types or numpy.ndarray')
            else:
                return _betainc(a, b, x/(1+x))

        return _betainc(a, b, randvar/(1+randvar))

    def mean(self) -> Union[float, str]:
        """
        Returns: Mean of the Beta prime distribution.
        """
        if self.beta > 1:
            return self.alpha/(self.beta-1)
        return "Undefined."

    def median(self) -> str:
        """
        Returns: Median of the Beta prime distribution.
        """
        # warning: not yet validated.
        return "Undefined."

    def mode(self) -> Union[float, str]:
        """
        Returns: Mode of the Beta prime distribution.
        """
        if self.alpha >= 1:
            return (self.alpha+1)/(self.beta+1)
        return 0.0

    def var(self) -> Union[float, str]:
        """
        Returns: Variance of the Beta prime distribution.
        """
        alpha = self.alpha
        beta = self.beta
        if beta > 2:
            return (alpha*(alpha+beta-1))/((beta-2)*(beta-1)**2)
        return "Undefined."

    def std(self) -> Union[float, str]:
        """
        Returns: Standard deviation of the Log logistic distribution
        """
        if self.var() == "Undefined.":
            return "Undefined."
        return _sqrt(self.var())

    def skewness(self) -> Union[float, str]:
        """
        Returns: Skewness of the Beta prime distribution.
        """
        alpha = self.alpha
        beta = self.beta
        if beta > 3:
            scale = (2*(2*alpha+beta-1))/(beta-3)
            return scale*_sqrt((beta-2)/(alpha*(alpha+beta-1)))
        return "Undefined."

    def kurtosis(self) -> str:
        """
        Returns: Kurtosis of the Beta prime distribution.
        """
        return "Undefined."

    def entropy(self) -> Union[float, str]:
        """
        Returns: differential entropy of the Beta prime distribution.

        Reference: Park, S.Y. & Bera, A.K.(2009). Maximum entropy autoregressive conditional heteroskedasticity model. Elsivier.
        link: http://wise.xmu.edu.cn/uploadfiles/paper-masterdownload/2009519932327055475115776.pdf
        """
        return "currently unsupported"


    def summary(self) -> Dict[str, Union[float, str]]:
        """
        Summary statistic regarding the Beta Prime distribution which contains the following parts of the distribution:
        (mean, median, mode, var, std, skewness, kurtosis).

        Returns:
            Dict[str, Union[float, str]]
        """
        return {
            'mean': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.var(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }
