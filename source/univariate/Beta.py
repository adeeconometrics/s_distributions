try:
    from scipy.special import beta as _beta, betainc as _betainc, digamma as _digamma
    import numpy as _np
    from typing import Union, Dict, List
    from math import sqrt as _sqrt, log as _log
    from univariate._base import BoundedInterval
except Exception as e:
    print(f"some modules are missing {e}")


class Beta(BoundedInterval):
    """
    This class contains methods concerning Beta Distirbution [#]_.

    .. math::
        \\text{Beta}(x; \\alpha, \\beta) = \\frac{x^{\\alpha-1}(1-x)^{\\beta-1}}{\\text{B}(\\alpha, \\beta)}

    Args:

        alpha(float): shape parameter where alpha > 0
        beta(float): shape parameter where beta > 0
        x(float): random variable where x is between 0 and 1

    Reference:
        .. [#] Wikipedia contributors. (2021, January 8). Beta distribution. https://en.wikipedia.org/w/index.php?title=Beta_distribution&oldid=999043368
    """

    def __init__(self, alpha: float, beta: float):
        if alpha < 0:
            raise ValueError(
                f'alpha parameter(shape) should be a positive number. Entered value:{alpha}')
        if beta < 0:
            raise ValueError(
                f'beta parameter(shape) should be a positive number. Entered value:{beta}')

        self.alpha = alpha
        self.beta = beta

    def pdf(self, x: Union[List[float], _np.ndarray, float]) -> Union[float, _np.ndarray]:
        """
        Args:
            x (Union[List[float], _np.ndarray, float]): random variables

        Raises:
            ValueError: when there exist a value x <= 0 or x <= 1
            TypeError: when parameter is not of type float | List[float] | numpy.ndarray    

        Returns:
            Union[float, _np.ndarray]: evaluation of pdf at x
        """
        a = self.alpha
        b = self.beta

        if isinstance(x, (_np.ndarray, List)):
            x = _np.fromiter(x, dtype=float)
            if _np.any(_np.logical_or(x<=0, x>=1)):
                raise ValueError('random variables should only be between 0 and 1')
            return (_np.power(x, a-1)*_np.power(1-x, b-1))/_beta(a, b)
   
        if type(x) is float:
            if x<=0 or x>=1:
                raise ValueError('random variables should only be between 0 and 1')
            return (pow(x, a-1)*pow(1-x, b-1))/_beta(a, b)  

        raise TypeError('parameter x is expected to be of type float | List[float] | numpy.ndarray')

    def cdf(self, x: Union[List[float], _np.ndarray, float]) -> Union[float, _np.ndarray]:
        """
        Args:
            x (Union[List[float], _np.ndarray]): random variable(s). 

        Returns:
            Union[float, _np.ndarray]: evaluation of cdf at x
        """
        a = self.alpha
        b = self.beta

        if isinstance(x, (_np.ndarray, List)):
            x = _np.fromiter(x,_np.float32)
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
        Returns:
            Dictionary of Beta distirbution moments. This includes standard deviation. 
        """
        return {
            'mean': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.var(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }
