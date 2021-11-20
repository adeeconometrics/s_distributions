try:
    from scipy.special import beta as _beta, betainc as _betainc
    import numpy as _np
    from typing import Union, Dict, List
    from math import sqrt as _sqrt
    from univariate._base import SemiInfinite
except Exception as e:
    print(f"some modules are missing {e}")


class BetaPrime(SemiInfinite):
    """
    This class contains methods concerning Beta prime Distirbution [#]_ .

    .. math:: 
        \\text{BetaPrime}(x;\\alpha,\\beta) = \\frac{x^{\\alpha -1}(1+x)^{-\\alpha -\\beta}}{\\text{B}(\\alpha ,\\beta )}

    Args:

        alpha(float): shape parameter where alpha > 0
        beta(float): shape parameter where beta > 0
        x(float): random variable where x >= 0

    Reference:
        .. [#] Wikipedia contributors. (2020, October 8). Beta prime distribution. https://en.wikipedia.org/w/index.php?title=Beta_prime_distribution&oldid=982458594
    """

    def __init__(self, alpha: float, beta: float):
        if alpha < 0:
            raise ValueError(
                'alpha parameter(shape) should be a positive number.')
        if beta < 0:
            raise ValueError(
                'beta parameter(shape) should be a positive number.')

        self.alpha = alpha
        self.beta = beta

    def pdf(self, x: Union[List[float], _np.ndarray, float]) -> Union[float, _np.ndarray]:
        """
        Args:
            x (Union[List[float], numpy.ndarray, float]): random variable(s)

        Raises:
            ValueError: when there exist a value of x less than 0

        Returns:
            Union[float, numpy.ndarray]: evaluation of pdf at x
        """
        a = self.alpha
        b = self.beta

        if isinstance(x, (_np.ndarray, List)):
            if not type(x) is _np.ndarray:
                x = _np.array(x)
            if _np.any(x < 0):
                raise ValueError('random variable should not be less then 0.')
            return _np.power(x, a-1)*_np.power(1+x, -a-b)/_beta(a, b)

        if x < 0:
            raise ValueError('random variable should not be less then 0.')
        return pow(x, a-1)*pow(1+x, -a-b)/_beta(a, b)

    def cdf(self, x: Union[List[float], _np.ndarray, float]) -> Union[float, _np.ndarray]:
        """
        Args:
            x (Union[List[float], numpy.ndarray, float]): data point(s) of interest

        Raises:
            ValueError: when there exist a value of x less than 0

        Returns:
            Union[float, numpy.ndarray]: evaluation of cdf at x
        """
        a = self.alpha
        b = self.beta

        if isinstance(x, (_np.ndarray, List)):
            if not type(x) is _np.ndarray:
                x = _np.array(x)
            if _np.any(x < 0):
                raise ValueError(
                    'evaluation of cdf is not supported for values less than 0')
            return _betainc(a, b, x/(1+x))

        return _betainc(a, b, x/(1+x))

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

    def mode(self) -> float:
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
        var = self.var()
        if type(var) is str:
            return "Undefined."
        return _sqrt(var)

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

    def entropy(self):
        """
        Returns: differential entropy of the Beta prime distribution.

        Reference: Park, S.Y. & Bera, A.K.(2009). Maximum entropy autoregressive conditional heteroskedasticity model. Elsivier.
        link: http://wise.xmu.edu.cn/uploadfiles/paper-masterdownload/2009519932327055475115776.pdf
        """
        return NotImplemented

    def summary(self) -> Dict[str, Union[float, str]]:
        """
        Returns:
            Dictionary of BetaPrime distirbution moments. This includes standard deviation. 
        """
        return {
            'mean': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.var(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }
