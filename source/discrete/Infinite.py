try:
    import numpy as np
    import scipy.special as ss
    from discrete._base import Base
    from typing import Union, Optional, Tuple, Dict, List
    import math as m
except Exception as e:
    print(f'some modules are missin{e}')


class Infinite(Base):
    """
    Description:
        Base class for probability tags.
    """

    def __init__(self):
        if type(self) is Infinite:
            raise TypeError('base class cannot be instantiated.')


class Poisson(Infinite):
    """
    This class contains methods for evaluating some properties of the poisson distribution. 
    As lambda increases to sufficiently large values, the normal distribution (λ, λ) may be used to 
    approximate the Poisson distribution [#]_ [#]_ [#]_.

    Use the Poisson distribution to describe the number of times an event occurs in a finite observation space.

    .. math:: \\text{Poisson}(x;\\lambda) = \\frac{\\lambda ^{x} e^{- \\lambda}}{x!}

    Args: 
        λ (float): expected rate if occurrences.
        x (int): number of occurrences.


    References:
        .. [#] Minitab (2019). Poisson Distribution. https://bityl.co/4uYc
        .. [#] Weisstein, Eric W. "Poisson Distribution." From MathWorld--A Wolfram Web Resource. https://mathworld.wolfram.com/PoissonDistribution.html.
        .. [#] Wikipedia contributors. (2020, December 16). Poisson distribution. https://en.wikipedia.org/w/index.php?title=Poisson_distribution&oldid=994605766
    """

    def __init__(self, λ: float):
        self.λ = λ

    def pmf(self, x: Union[List[int], int, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Args:
            x (Union[List[int], int, np.ndarray]): random variable(s)

        Raises:
            TypeError: when types not of type integer
            ValueError: when x is less than 0

        Returns:
            Union[float, np.ndarray]: evaluation of pmf at x
        """

        if isinstance(x, (List, np.ndarray)):
            if not type(x) is np.ndarray:
                x = np.array(x)
            if not np.issubdtype(x[0], np.integer):
                raise TypeError('parameter x must be a positive integer')
            return (np.power(self.λ, x) * np.exp(-self.λ)) / np.vectorize(m.factorial, otypes=[object])(x)

        if x < 0:
            raise ValueError('parameter x must be a positive integer')
        return (pow(self.λ, x) * m.exp(-self.λ)) / m.factorial(x)

    def cdf(self, x: Union[List[int], int, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Args:
            x (Union[List[int], int, np.ndarray]): data point(s) of interest

        Raises:
            TypeError: when types are not of type integer
            ValueError: when x is les than 0

        Returns:
            Union[float, np.ndarray]: evaluation of cdf at x
        """
        λ = self.λ

        if isinstance(x, (List, np.ndarray)):
            if not type(x) is np.ndarray:
                x = np.array(x)
            if not np.issubdtype(x[0], np.integer):
                raise TypeError('parameter x must be a positive integer')
            return ss.gammainc(m.floor(x + 1), λ) / np.vectorize(m.factorial, otypes=[object])(np.floor(x))

        if x < 0:
            raise ValueError('parameter x must be a positive integer')
        return ss.gammainc(m.floor(x + 1), λ) / m.factorial(m.floor(x))

    def mean(self) -> float:
        """
        Returns: 
            the mean of Poisson Distribution.
        """
        return self.λ

    def median(self) -> float:
        """
        Returns: 
            the median of Poisson Distribution.
        """
        λ = self.λ
        return λ + 1 / 3 - (0.02 / λ)

    def mode(self) -> Tuple[int, int]:
        """
        Returns: 
            the mode of Poisson Distribution.
        """
        λ = self.λ
        return m.ceil(λ) - 1, m.floor(λ)

    def var(self) -> float:
        """
        Returns: 
            the variance of Poisson Distribution.
        """
        return self.λ

    def skewness(self) -> float:
        """
        Returns: 
            the skewness of Poisson Distribution.
        """
        return pow(self.λ, -0.5)

    def kurtosis(self) -> float:
        """
        Returns: 
            the kurtosis of Poisson Distribution.
        """
        return 1/self.λ

    def summary(self) -> Dict[str, Union[float, Tuple[int, int]]]:
        """
        Returns:
            Dictionary of Poisson distirbution moments. This includes standard deviation. 
        """
        return {
            'mean': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.var(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }


class Zeta(Infinite):
    """
    This class contains methods concerning the Zeta Distribution [#]_ [#]_.

    .. math:: \\text{Zeta}(x;s) =\\frac{\\frac{1}{x^s}}{\\zeta(s)}

    Args:
        - s (float): main parameter
        - x (int): support parameter

    References:
        .. [#] Wikipedia contributors. (2020, November 6). Zeta distribution. In Wikipedia, The Free Encyclopedia. Retrieved 10:24, December 26, 2020, from https://en.wikipedia.org/w/index.php?title=Zeta_distribution&oldid=987351423
        .. [#] The Zeta Distribution. (2021, February 3). https://stats.libretexts.org/@go/page/10473
    """

    def __init__(self, s: float):
        self.s = s

    def pmf(self, x: Union[List[int], np.ndarray, int]) -> Union[float, np.ndarray]:
        """
        Args:
            x (Union[List[int], np.ndarray, int]): random variable(s)

        Raises:
            TypeError: when types are not of type integer

        Returns:
            Union[float, np.ndarray]: evaluation of pmf at x
        """
        s = self.s

        if isinstance(x, (List, np.ndarray)):
            if not type(x) is np.ndarray:
                x = np.array(x)
            if not np.issubdtype(x[0], np.integer):
                raise TypeError('random variables must be of type integer')

        return (1/x**s)/ss.zeta(s)

    def cdf(self, x: List[int]) -> Union[int, float, List[int]]:
        """
        Args:
            x (List[int]): random variables.

        Returns:
            Union[int, float, List[int]]: evaluation of cdf at x. Currently NotImplemented
        """
        return NotImplemented

    def mean(self) -> Union[str, float]:
        """
        Returns: 
            mean of Zeta distribution
        """
        s = self.s
        if s > 2:
            return ss.zeta(s - 1) / ss.zeta(s)
        return "undefined"

    def median(self) -> str:
        """
        Returns: 
            undefined.
        """
        return "undefined"

    def mode(self) -> int:
        """
        Returns: 
            mode of Zeta distribution
        """
        return 1

    def var(self) -> Union[str, float]:
        """
        Returns: 
            the variance of Zeta Distribution. Returns undefined if s <= 3.
        """
        s = self.s
        if s > 3:
            _x0 = ss.zeta(s)
            return (ss.zeta(s-2)/_x0) - (ss.zeta(s-1)/ss.zeta(s))**2
        return "undefined"

    def std(self) -> Union[str, float]:
        """
        Returns:
            the standard deviation of Zeta Distribution. Returns undefined if variance is undefined.
        """
        s = self.s
        if s > 3:
            _x0 = ss.zeta(s)
            return m.sqrt((ss.zeta(s-2)/_x0) - (ss.zeta(s-1)/ss.zeta(s))**2)
        return "undefined"

    def skewness(self) -> Union[str, float]:
        """
        Returns: 
            the skewness of Zeta Distribution.
        """
        s = self.s
        if s <= 4:
            return "undefined"
        _x0 = ss.zeta(s-2)*ss.zeta(s)
        _x1 = ss.zeta(s-1)
        return (ss.zeta(s-3)*ss.zeta(s)**2 - 3*_x1*_x0 + 2*_x1**3) / pow(_x0-_x1**2, 3/2)

    def kurtosis(self) -> Union[str, float]:
        """
        Returns: 
            the kurtosis of Zeta Distribution.
        """
        s = self.s
        if s <= 5:
            return "undefined"

        _x0 = ss.zeta(s-2)
        _x1 = ss.zeta(s)
        _x3 = ss.zeta(s-1)

        scale = 1/pow(_x0*_x1 - _x3**2, 2)
        numerator = (ss.zeta(s-4)*_x1**3) - (4*_x3*ss.zeta(s-3)
                                             * _x1**2) + (6*_x3**2*_x0*_x1-3*_x3**4)
        return scale*numerator

    def summary(self) -> Dict[str, Union[float, int, str]]:
        """
        Returns:
            Dictionary of Zeta distirbution moments. This includes standard deviation. 
        """
        return {
            'mean': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.var(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }


# class CMP(Infinite):
#     ...

# value checking on rv

class Borel(Infinite):
    """
    This class contains methods concerning the Borel Distribution [#]_.

    .. math::
        \\text{Borel}(x;\\mu) = \\frac{e^{-\\mu x}(\\mu x)^{x-1}}{x!}

    Args:
        mu (float): mu parameter :math:`\\mu \\in [0,1]`.
        x (int): random variables

    Reference: 
        .. [#] Wikipedia Contributors (2021). Borel Distribution. https://en.wikipedia.org/wiki/Borel_distribution

    """

    def __init__(self, mu: float):
        if mu < 0 or mu > 1:
            raise ValueError('mu parameter must belong in the domain [0,1]')

        self.mu = mu

    def pmf(self, x: Union[List[int], int, np.ndarray]) -> Union[float, np.ndarray]:
        mu = self.mu
        if isinstance(x, (List, np.ndarray)):
            if not type(x) is np.ndarray:
                x = np.array(x)
            if not np.issubdtype(x[0], np.integer):
                raise TypeError(
                    'parameter x are expected to be of type integer.')
            return np.exp(-mu*x)*np.power(mu*x, x-1)/np.vectorize(m.factorial, otypes=[object])(x)

        return m.exp(-mu*x)*pow(x*mu, x-1)/m.factorial(x)

    def mean(self) -> float:
        return 1/(1-self.mu)

    def var(self) -> float:
        return self.mu/pow(1-self.mu, 3)

    def std(self) -> float:
        return m.sqrt(self.mu/pow(1-self.mu, 3))

    def summary(self) -> Dict[str, float]:
        """
        Returns:
            Dictionary of Borel distirbution moments. This includes standard deviation. 
        """
        return {
            'mean': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.var(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }


# class Benford(Infinite):
#     ...

# value checking on rv
class Logarithmic(Infinite):
    """This class contains methods concerning Logarithmic distirbution [#]_.

    .. math::
        \\text{Logarithmic}(x;p) = \\frac{-1}{\\ln(1-p)} \\ \\frac{p^k}{k}

    Args:
        p (float): p parameter
        x (int): random variable

    Reference:
        .. [#] Wikipedia Contributors (2020). Logarithmic distirbution. https://en.wikipedia.org/wiki/Logarithmic_distribution.
    """

    def __init__(self, p: float):
        if p <= 0 or p >= 1:
            raise ValueError(
                'parameter p is expected to be defined in the domain (0,1)')

        self.p = p

    def pmf(self, x: Union[List[int], int, np.ndarray]) -> Union[float, np.ndarray]:
        p = self.p
        x0 = -1/m.log(1-p)

        if isinstance(x, (List, np.ndarray)):
            if not type(x) is np.ndarray:
                x = np.array(x)
            if np.issubdtype(x[0], np.integer):
                raise TypeError(
                    'parameter x is expected to be of type integer')

            return x0*np.power(p, x)/x

        if not type(x) is int:
            raise TypeError('parameter x is expected to be of type integer')
        return x0*pow(p, x)/x

    def cdf(self, x: Union[List[int], int, np.ndarray]) -> Union[float, np.ndarray]:
        p = self.p

        if isinstance(x, (List, np.ndarray)):
            if not type(x) is np.ndarray:
                x = np.array(x)
            if np.issubdtype(x[0], np.integer):
                raise TypeError(
                    'parameter x is expected to be of type integer')
        else:
            if not type(x) is int:
                raise TypeError(
                    'parameter x is expected to be of type integer')

        return 1 + ss.betainc(p, x+1, 0)/m.log(1-p)

    def mean(self) -> float:
        p = self.p
        return (-1/m.log(1-p))*p/(1-p)

    def mode(self) -> float:
        return 1.0

    def var(self) -> float:
        p = self.p
        x0 = p**2 + p*m.log(1-p)
        x1 = pow(1-p, 2)*pow(m.log(1-p), 2)
        return - x0/x1

    def std(self) -> float:
        return m.sqrt(self.var())

    def summary(self) -> Dict[str, float]:
        """
        Returns:
            Dictionary of Logarithmic distirbution moments. This includes standard deviation. 
        """
        return {
            'mean': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.var(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }

# class Skellam(Infinite):
#     ...

# value checking on rv


class YulleSimon(Infinite):
    """
    This class contains methods concerning the Yulle-Simon Distribution [#]_.

    .. math:: 
        \\text{YulleSimon}(x;\\rho) = \\rho\\text{B}(x,\\rho+1)

    Args:
        shape (float): shape parameter :math:`\\rho > 0`
        x (float): random variables

    Reference: 
        .. [#] Wikipedia Contributors (2021). Yulle-Simon distribution. https://en.wikipedia.org/wiki/Yule%E2%80%93Simon_distribution.
    """

    def __init__(self, shape: float):
        if shape <= 0:
            raise ValueError('shape parameter must be a positive real number')

        self.shape = shape

    def pmf(self, x: Union[List[int], int, np.ndarray]) -> Union[float, np.ndarray]:

        shape = self.shape
        if isinstance(x, (List, np.ndarray)):
            if not type(x) is np.ndarray:
                x = np.array(x)
            if np.issubdtype(x[0], np.integer):
                raise TypeError(
                    'parameter x is expected to be of type integer')

        else:
            if not type(x) is int:
                raise TypeError(
                    'parameter x is expected to be of type integer')

        return shape*ss.beta(x, shape+1)

    def cdf(self, x: Union[List[int], int, np.ndarray]) -> Union[float, np.ndarray]:
        shape = self.shape
        if isinstance(x, (List, np.ndarray)):
            if not type(x) is np.ndarray:
                x = np.array(x)
            if np.issubdtype(x[0], np.integer):
                raise TypeError(
                    'parameter x is expected to be of type integer')

        else:
            if not type(x) is int:
                raise TypeError(
                    'parameter x is expected to be of type integer')

        return (1-x)*ss.beta(x, shape+1)

    def mean(self) -> Union[float, str]:
        shape = self.shape

        if shape > 1:
            return shape/(shape-1)

        return 'Undefined'

    def mode(self) -> float:
        return 1.0

    def var(self) -> Union[float, str]:
        shape = self.shape
        if shape > 2:
            return shape**2/(pow(shape-1, 2)*(shape-2))
        return 'Undefined'

    def std(self) -> Union[float, str]:
        shape = self.shape
        if shape > 2:
            return m.sqrt(shape**2/(pow(shape-1, 2)*(shape-2)))
        return 'Undefined'

    def skewness(self) -> Union[float, str]:
        shape = self.shape
        if shape > 3:
            return pow(shape+1, 2)*m.sqrt(shape-2)/((shape-3)*shape)
        return 'Undefined'

    def kurtosis(self) -> Union[float, str]:
        shape = self.shape
        if shape > 4:
            x0 = (11*pow(shape, 3) - 49*shape - 22)/((shape-4)*(shape-3)*shape)
            return shape + 3 + x0
        return 'Undefined'

    def summary(self) -> Dict[str, Union[float, str]]:
        """
        Returns:
            Dictionary of Borel distirbution moments. This includes standard deviation. 
        """
        return {
            'mean': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.var(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }


class GaussKuzmin(Infinite):
    """This class contains methods concerning the Gauss-Kuzmin distribution [#]_ .

    .. math:: 
        \\text{GaussKuzmin}(x) = -\\log_2 \\Big[ 1- \\frac{1}{(k+1)^2}\\Big]

    Args:
        x (int): random variables :math:`x \\in (0,\\inf]`

    Reference: 
        .. [#] Wikipedia Contributors (2021). Gauss-Kuzmin distribution. https://en.wikipedia.org/wiki/Gauss%E2%80%93Kuzmin_distribution.

    """

    def pmf(self, x: Union[List[int], int, np.ndarray]) -> Union[float,
                                                                 np.ndarray]: ...

    def cdf(self, x: Union[List[int], int, np.ndarray]) -> Union[float,
                                                                 np.ndarray]: ...

    def mean(self) -> float:
        return float('inf')

    def median(self) -> float:
        return 2.0

    def mode(self) -> float:
        return 1.0

    def var(self) -> float:
        return float('inf')

    def std(self) -> float:
        return float('inf')

    def skewness(self) -> str:
        return'Undefined'

    def kurtosis(self) -> str:
        return 'Undefined'

    def entropy(self) -> float:
        """ Reference: 
        N. Blachman (1984). The Continued fraction as an information source (Corresp.). 
        IEEE Transactions on Information Theor (Volume: 30, Issue: 4). DOI:  10.1109/TIT.1984.1056924.
        """
        return 3.432527514776

    def summary(self) -> Dict[str, Union[float, str]]:
        """
        Returns:
            Dictionary of Poisson distirbution moments. This includes standard deviation. 
        """
        return {
            'mean': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.var(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }
