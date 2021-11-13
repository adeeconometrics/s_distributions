try:
    from scipy.special import erf as _erf
    from numpy import euler_gamma as _euler_gamma
    import numpy as _np
    from typing import Union, Dict, List
    from math import sqrt as _sqrt, log as _log, pi as _pi, exp as _exp
    from univariate._base import SemiInfinite
except Exception as e:
    print(f"some modules are missing {e}")


class MaxwellBoltzmann(SemiInfinite):
    """
    This class contains methods concerning Maxwell-Boltzmann Distirbution [#]_.

    .. math::
        \\text{MaxwellBoltzmann}(x;a) = \\sqrt{\\frac{2}{\\pi}} \\frac{x^2 \\exp{-x^2/(2a^2)}}{a^3}

    Args:

        a(int): parameter where a > 0
        x(float): random variable where x >= 0

    Reference:
        .. [#] Wikipedia contributors. (2021, January 12). Maxwell–Boltzmann distribution. https://en.wikipedia.org/w/index.php?title=Maxwell%E2%80%93Boltzmann_distribution&oldid=999883013
    """

    def __init__(self, a: int):
        if a < 0:
            raise ValueError(
                'parameter a should be a positive number. Entered value:{}'.format(a))
        if type(a) is not int:
            raise TypeError('parameter should be in type int')

        self.a = a

    def pdf(self, x: Union[List[float], _np.ndarray, float]) -> Union[float, _np.ndarray]:
        """
        Args:
            x (Union[List[float], _np.ndarray, float]): random variable(s)

        Raises:
            ValueError: when there exist a value of x less than 0

        Returns:
            Union[float, _np.ndarray]: evaluation of pdf at x
        """
        a = self.a

        if isinstance(x, (_np.ndarray, List)):
            x = _np.array(x)
            if _np.any(x < 0):
                raise ValueError('random values must not be lesser than 0')
            return _sqrt(2/_pi)*(x**2*_np.exp(-x**2/(2*a**2)))/a**3

        if x < 0:
            raise ValueError('random values must not be lesser than 0')
        return _sqrt(2/_pi)*(x**2*_exp(-x**2/(2*a**2)))/a**3

    def cdf(self, x: Union[List[float], _np.ndarray, float]) -> Union[float, _np.ndarray]:
        """
        Args:
            x (Union[List[float], _np.ndarray, float]): data point(s) or interest

        Returns:
            Union[float, _np.ndarray]: evaluation of cdf at x
        """
        a = self.a

        if isinstance(x, (_np.ndarray, List)):
            x = _np.array(x)
            x0 = _np.power(x, 2)
            return _erf(x/(_sqrt(2)*a))-_sqrt(2/_pi)*(x0*_np.exp(-x0/(2*a**2)))/(a)

        return _erf(x/(_sqrt(2)*a)) - _sqrt(2/_pi)*(x**2*_exp(-x**2/(2*a**2)))/(a)

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

    def summary(self) -> Dict[str, Union[float, str]]:
        """
        Returns:
            Dictionary of Maxwell-Boltzmann distirbution moments. This includes standard deviation. 
        """
        return {
            'mean': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.var(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }
