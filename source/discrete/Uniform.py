try:
    import numpy as _np
    from typing import Union, Tuple, Dict, List
    from math import floor as _floor
    from discrete._base import Base
except Exception as e:
    print(f"some modules are missing {e}")


class Uniform(Base):
    """
    This contains methods for finding the probability mass function and 
    cumulative distribution function of Uniform distribution. Incudes scatter plot [#]_. 

    .. math:: \\text{Uniform} (a,b) = {\\begin{cases}{\\frac {1}{b-a}}&\\mathrm {for} \ a\leq x\leq b,\\\[8pt]0&\mathrm {for} \ x<a\ \mathrm {or} \ x>b\\end{cases}}

    Args: 
        data (int): sample size

    Reference:
        .. [#] NIST/SEMATECH e-Handbook of Statistical Methods (2012). Uniform Distribution. Retrieved from http://www.itl.nist.gov/div898/handbook/, December 26, 2020.
    """

    def __init__(self, a: int, b: int):
        if type(a) and type(b) is not int:
            raise TypeError('parameter a and b should be of type integer')

        self.a = a
        self.b = b
        self.n = abs(b-a+1)

    def pmf(self, x: Union[List[int], _np.ndarray, int]) -> Union[float,  _np.ndarray]:
        """
        Args:
            x (Union[List[int], _np.ndarray, int]): random variable(s)

        Returns:
            Union[float,  _np.ndarray]: evaluation of pmf at x
        """

        if isinstance(x, (List, _np.ndarray)):
            x = _np.empty(len(x))
            x[:] = 1/self.n
            return x
        return 1 / self.n

    def cdf(self, x: Union[List[int], _np.ndarray, int]) -> Union[float,  _np.ndarray]:
        """
        Args:
            x (Union[List[int], _np.ndarray, int]): data point(s)

        Returns:
            Union[float,  _np.ndarray]: evaluation of cdf at x
        """

        a, b, n = self.a, self.b, self.n

        if isinstance(x, (List, _np.ndarray)):
            if not type(x) is _np.ndarray:
                x = _np.array(x)
            if not _np.issubdtype(x[0], _np.integer):
                raise TypeError('random variables must be of type integer')
            return _np.piecewise(x, [x < a, (x >= a) & (x <= b), x > b], [0.0, lambda x: (_np.floor(x-a) + 1)/n, 1.0])
        return (_floor(x-a) + 1)/n if x >= a and x <= b else (0.0 if x < a else 1.0)

    def mean(self) -> float:
        """
        Returns: 
            the mean of Uniform Distribution.
        """
        return (self.a + self.b) / 2

    def median(self) -> float:
        """
        Returns: 
            the median of Uniform Distribution.
        """
        return (self.a + self.b) / 2

    def mode(self) -> Tuple[int, int]:
        """
        Returns: 
            the mode of Uniform Distribution.
        """
        return (self.a, self.b)

    def var(self) -> float:
        """
        Returns: 
            the variance of Uniform Distribution.
        """
        return (self.b - self.a)**2 / 12

    def skewness(self) -> int:
        """
        Returns: 
            the skewness of Uniform Distribution.
        """
        return 0

    def kurtosis(self) -> float:
        """
        Returns: 
            the kurtosis of Uniform Distribution.
        """
        return -6 / 5

    def summary(self) -> Dict[str, Union[float, Tuple[int, int]]]:
        """
        Returns:
            Dictionary of Uniform distirbution moments. This includes standard deviation. 
        """
        return {
            'mean': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.var(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }
