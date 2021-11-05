try:
    import numpy as _np
    from typing import Union, Tuple, Dict, List
    from univariate._base import BoundedInterval
except Exception as e:
    print(f"some modules are missing {e}")


class Trapezoidal(BoundedInterval):
    """
    This class contains methods concerning Trapezoidal Distirbution.
    Args:

        a(float | a<d): lower bound
        b(float | a≤b<c): level start
        c(float | b<c≤d): level end
        d(float | c≤d): upper bound
        randvar(float | a≤randvar≤d): random variable   

    Reference:
    - Wikipedia contributors. (2020, April 11). Trapezoidal distribution. In Wikipedia, The Free Encyclopedia.
    Retrieved 06:06, December 30, 2020, from https://en.wikipedia.org/w/index.php?title=Trapezoidal_distribution&oldid=950241388
    """

    def __init__(self, a: float, b: float, c: float, d: float, randvar: float):
        if a > d:
            raise ValueError(
                'lower bound(a) should be less than upper bound(d).')
        if a > b or b >= c:
            raise ValueError(
                'lower bound(a) should be less then or equal to level start (b) where (b) is less than level end(c).')
        if b >= c or c > d:
            raise ValueError(
                'level start(b) should be less then level end(c) where (c) is less then or equal to upper bound (d).')
        if c > d:
            raise ValueError(
                'level end(c) should be less than or equal to upper bound(d)')

        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.randvar = randvar

    def pdf(self, x: Union[List[float], _np.ndarray] = None) -> Union[float, List[float]]:
        """
        Args:

            x (List[float], numpy.ndarray): random variable or list of random variables

        Returns:
            either probability density evaluation for some point or plot of Trapezoidal distribution.
        """ 
        a,b,c,d = self.a, self.b, self.c, self.d
        randvar = self.randvar

        def __generator(a:float, b:float, c:float, d:float, x:float) -> float:
            if a <= x and x < b:
                return 2/(d+c-a-b) * (x-a)/(b-a)
            if b <= x and x < c:
                return 2/(d+c-a-b)
            if c <= x and x <= d:
                return (2/(d+c-a-b))*(d-x)/(d-c)

        if x is not None:
            if not isinstance(x, (_np.ndarray, List)):
                raise TypeError(f'parameter x only accepts List types or numpy.ndarray')
            else:
                return [__generator(a,b,c,d,i) for i in x]
        return __generator(a,b,c,d,radvar)

    def cdf(self, x: Union[List[float], _np.ndarray] = None) -> Union[float, List[float]]:
        """
        Args:

            x (List[float], numpy.ndarray): random variable or list of random variables

        Returns:
            either cumulative distribution evaluation for some point or plot of Trapezoidal distribution.
        """ 
        def __generator(a:float, b:float, c:float, d:float, x:float) -> float:
            if a <= x and x < b:
                return (x-a)**2/((b-a)*(d+c-a-b))
            if b <= x and x < c:
                return (2*x-a-b)/(d+c-a-b)
            if c <= x and x <= d:
                return 1 - (d-x)**2/((d+c-a-b)*(d-c))

        if x is not None:
            if not isinstance(x, (_np.ndarray, List)):
                raise TypeError(f'parameter x only accepts List types or numpy.ndarray')
            else:
                return [__generator(a,b,c,d,i) for i in x]
        return __generator(a,b,c,d,radvar)


    def mean(self) -> float:
        """
        Returns: Mean of the Trapezoidal distribution.
        """
        a = self.a
        b = self.b
        c = self.c
        d = self.d

        return 1/(3*(d+c-b-a)) * ((d**3 - c**3)/(d-c) - (b**3 - a**3)/(b-a))

    def var(self) -> float:
        """
        Returns: Variance of the Trapezoidal distribution. Currently Unsupported.
        """
        a = self.a
        b = self.b
        c = self.c
        d = self.d

        mean = 1/(3*(d+c-b-a)) * ((d**3 - c**3)/(d-c) - (b**3 - a**3)/(b-a))
        return 1/(6*(d+c-b-a)) * ((d**4 - c**4)/(d-c) - (b**4 - a**4)/(b-a)) - pow(mean, 2)


    def summary(self) -> Dict[str, Union[float, Tuple[float]]]:
        """
        Summary statistic regarding the Trapezoidal distribution which contains the following parts of the distribution:
        (mean, median, mode, var, std, skewness, kurtosis).

        Returns:
            Dict[str, Union[float, Tuple[float]]]: [description]
        """
        return {
            'mean': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.var(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }

