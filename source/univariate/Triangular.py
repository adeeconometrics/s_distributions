try:
    import numpy as _np
    from typing import Union, Dict, List, Optional
    from math import sqrt as _sqrt, log as _log
    from univariate._base import BoundedInterval
except Exception as e:
    print(f"some modules are missing {e}")


class Triangular(BoundedInterval):
    """
    This class contains methods concerning Triangular Distirbution.
    Args:

        a(float): lower limit
        b(float | a<b): upper limit
        c(float| a≤c≤b): mode
        randvar(float | a≤randvar≤b): random variable

    Reference:
    - Wikipedia contributors. (2020, December 19). Triangular distribution. In Wikipedia, The Free Encyclopedia.
    Retrieved 05:41, December 30, 2020, from https://en.wikipedia.org/w/index.php?title=Triangular_distribution&oldid=995101682
    """

    def __init__(self, a: float, b: float, c: float, randvar: float):
        if a > b:
            raise ValueError(
                'lower limit(a) should be less than upper limit(b).')
        if a > c and c > b:
            raise ValueError(
                'lower limit(a) should be less than or equal to mode(c) where c is less than or equal to upper limit(b).')
        if a > randvar and randvar > b:
            raise ValueError(
                f'random variable is bounded between a: {a} and b: {b}')

        self.a = a
        self.b = b
        self.c = c
        self.randvar = randvar

    def pdf(self, x: Union[List[float], _np.ndarray] = None) -> Union[float, List]:
        """
        Args:

            x (List[float], numpy.ndarray): random variable or list of random variables

        Returns:
            either probability density evaluation for some point or plot of Triangular distribution.
        """ 
        a,b,c,d = self.a, self.b, self.c, self.d
        randvar = self.randvar

        def __generator(a:float, b:float, c:float, x:float)->float:
            if x < a:
                return 0.0
            if a <= x and x < c:
                return (2*(x-a))/((b-a)*(c-a))
            if x == c:
                return 2/(b-a)
            if c < x and x <= b:
                return (2*(b-x))/((b-a)((b-c)))
            if b < x:
                return 0.0

        if x is not None:
            if not isinstance(x, (_np.ndarray, List)):
                raise TypeError(f'parameter x only accepts List types or numpy.ndarray')
            else:
                return [__generator(a,b,c,d,i) for i in x]
        return __generator(a,b,c,d,randvar)

    def cdf(self, x: Union[List[float], _np.ndarray] = None) -> Union[float, List]:
        """
        Args:

            x (List[float], numpy.ndarray): random variable or list of random variables

        Returns:
            either cumulative density evaluation for some point or plot of Triangular distribution.
        """ 
        a,b,c,d = self.a, self.b, self.c, self.d
        randvar = self.randvar

        def __generator(a:float, b:float, c:float, x:float)->float:
            if x <= a:
                return 0.0
            if a < x and x <= c:
                return pow(x-a, 2)/((b-a)*(c-a))
            if c < x and x < b:
                return 1 - pow(b-x, 2)/((b-c)*(b-c))
            if b <= x:
                return 1.0

        if x is not None:
            if not isinstance(x, (_np.ndarray, List)):
                raise TypeError(f'parameter x only accepts List types or numpy.ndarray')
            else:
                return [__generator(a,b,c,d,i) for i in x]
        return __generator(a,b,c,d,randvar)


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
        Summary statistic regarding the Triangular distribution which contains the following parts of the distribution:
        (mean, median, mode, var, std, skewness, kurtosis).

        Returns:
            Dict[str, float]
        """
        return {
            'mean': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.var(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }

