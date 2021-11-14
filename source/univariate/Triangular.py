try:
    import numpy as _np
    from typing import Union, Dict, List
    from math import sqrt as _sqrt, log as _log
    from univariate._base import BoundedInterval
except Exception as e:
    print(f"some modules are missing {e}")


class Triangular(BoundedInterval):
    """
    This class contains methods concerning Triangular Distirbution [#]_.

    Args:

        a(float): lower limit parameter
        b(float): upper limit parameter where a < b
        c(float): mode parameter where a <= c <= b
        randvar(float): random variable where a <= x <= b

    Reference:
        .. [#] Wikipedia contributors. (2020, December 19). Triangular distribution. https://en.wikipedia.org/w/index.php?title=Triangular_distribution&oldid=995101682
    """

    def __init__(self, a: float, b: float, c: float):
        if a > b:
            raise ValueError(
                'lower limit(a) should be less than upper limit(b).')
        if a > c and c > b:
            raise ValueError(
                'lower limit(a) should be less than or equal to mode(c) where c is less than or equal to upper limit(b).')
        self.a = a
        self.b = b
        self.c = c

    def pdf(self, x: Union[List[float], _np.ndarray, float]) -> Union[float, _np.ndarray]:
        """
        Args:
            x (Union[List[float], numpy.ndarray, float]): random variable(s)

        Raises:
            ValueError: when there exist a value of a > x or x > b 

        Returns:
            Union[float, numpy.ndarray]: evaluation of pdf at x
        """
        a, b, c = self.a, self.b, self.c

        def __generator(a: float, b: float, c: float, x: float) -> float:
            if x < a:
                return 0.0
            if a <= x and x < c:
                return (2*(x-a))/((b-a)*(c-a))
            if x == c:
                return 2/(b-a)
            if c < x and x <= b:
                return 2*(b-x)/((b-a)*((b-c)))
            if b < x:
                return 0.0

        if isinstance(x, (_np.ndarray, List)):
            if not type(x) is _np.ndarray:
                x = _np.array(x)
            if _np.any(_np.logical_or(a > x, x > b)):
                raise ValueError(
                    'all random variables are expected to be between a and b parameters')
            return _np.vectorize(__generator)(a, b, c, x)

        if a > x or x > b:
            raise ValueError(
                'all random variables are expected to be between a and b parameters')

        return __generator(a, b, c, x)

    def cdf(self, x: Union[List[float], _np.ndarray, float]) -> Union[float, _np.ndarray]:
        """
        Args:
            x (Union[List[float], numpy.ndarray, float]): data point(s) of interest

        Returns:
            Union[float, numpy.ndarray]: evaluation fo cdf at x
        """
        a, b, c = self.a, self.b, self.c

        def __generator(a: float, b: float, c: float, x: float) -> float:
            if x <= a:
                return 0.0
            if a < x and x <= c:
                return pow(x-a, 2)/((b-a)*(c-a))
            if c < x and x < b:
                return 1 - pow(b-x, 2)/((b-c)*(b-c))
            if b <= x:
                return 1.0

        if isinstance(x, (_np.ndarray, List)):
            if not type(x) is _np.ndarray:
                x = _np.array(x)
            return _np.vectorize(__generator)(a, b, c, x)

        return __generator(a, b, c, x)

    def mean(self) -> float:
        """
        Returns: Mean of the Triangular distribution.
        """
        return (self.a+self.b+self.c)/3

    def median(self) -> float:
        """
        Returns: Median of the Triangular distribution.
        """
        a = self.a
        b = self.b
        c = self.c
        if c >= (a+b)/2:
            return a + _sqrt(((b-a)*(c-a))/2)
        if c <= (a+b)/2:
            return b + _sqrt((b-a)*(b-c)/2)

    def mode(self) -> float:
        """
        Returns: Mode of the Triangular distribution.
        """
        return self.c

    def var(self) -> float:
        """
        Returns: Variance of the Triangular distribution.
        """
        a = self.a
        b = self.b
        c = self.c
        return (1/18)*(pow(a, 2)+pow(b, 2)+pow(c, 2)-a*b-a*c-b*c)

    def std(self) -> float:
        """
        Returns: Standard deviation of the Triangular distribution.
        """
        return _sqrt(self.var())

    def skewness(self) -> float:
        """
        Returns: Skewness of the Triangular distribution.
        """
        a = self.a
        b = self.b
        c = self.c
        return _sqrt(2)*(a+b-2*c) * ((2*a-b-c)*(a-2*b+c)) / \
            (5*pow(a**2+b**2+c**2-a*b-a*c-b*c, 3/2))

    def kurtosis(self) -> float:
        """
        Returns: Kurtosis of the Triangular distribution.
        """
        return -3/5

    def entropy(self) -> float:
        """
        Returns: differential entropy of the Triangular distribution.

        Reference: Park, S.Y. & Bera, A.K.(2009). Maximum entropy autoregressive conditional heteroskedasticity model. Elsivier.
        link: http://wise.xmu.edu.cn/uploadfiles/paper-masterdownload/2009519932327055475115776.pdf
        """
        return 0.5 + _log((self.b-self.a)*0.5)

    def summary(self) -> Dict[str, float]:
        """
        Returns:
            Dictionary of Triangular distirbution moments. This includes standard deviation. 
        """
        return {
            'mean': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.var(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }
