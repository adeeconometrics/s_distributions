try:
    import numpy as _np
    from math import sqrt as _sqrt, ceil as _ceil, floor as _floor, log2 as _log2
    from typing import Union, Tuple, Dict, List
    from _base import Finite
except Exception as e:
    print(f"some modules are missing {e}")


class Geometric(Finite):
    """
    This class contains functions for finding the probability mass function and 
    cumulative distribution function for geometric distribution. We consider two definitions 
    of the geometric distribution: one concerns itself to the number of X of Bernoulli trials
    needed to get one success, supported on the set {1,2,3,...}. The second one concerns with 
    Y=X-1 of failures before the first success, supported on the set {0,1,2,3,...} [#]_ [#]_. 

    .. math:: \\text{Geometric}_1(x;p) = (1-p)^{x-1}p
    .. math:: \\text{Geometric}_2(x;p) = (1-p)^{x}p

    Args:

        p (float): success probability for each trial. Where 0 <= p <= 1.
        x (int): number of successes 

    References:
        .. [#] Weisstein, Eric W. "Geometric Distribution." From MathWorld--A Wolfram Web Resource. https://mathworld.wolfram.com/GeometricDistribution.html
        .. [#] Wikipedia contributors. (2020, December 27). Geometric distribution. In Wikipedia, The Free Encyclopedia. Retrieved 12:05, December 27, 2020, from https://en.wikipedia.org/w/index.php?title=Geometric_distribution&oldid=996517676

    Note: Geometric distribution can be configured based through `_type` parameter in `pmf`, `cdf` and moments of the distribution, including the `std`. 
    The default type is `_type='first'`, or :math:`\\text{Geometric_1}(x;p)`.
    """

    def __init__(self, p:float):
        if p < 0 or p > 1:
            raise ValueError('parameter p is constrained at')

        self.p = p

    def pmf(self, x: Union[List[int], int], _type: str = 'first') -> Union[List[float], float]:
        """
        Args:         
            type (keyvalue ∈[fist, second]): defaults to first. Reconfigures the type of distribution.
            x (Union[List[int], int]): list of random variables

        Returns: 
            probability mass evaluation of geometric distribution to some point specified by k or a 
            list of its corresponding value specified by the parameter x.

        Note: there are two configurations of pmf. 
        """
        p = self.p
        try:
            generator = {'first': lambda p,k: pow(1-p, k-1)*p, 
                        'second': lambda p,k: pow(1-p, k)*p}
            if isinstance(x, List):
                if any(type(i) is not int for i in x):
                    raise TypeError('parameter k must be of type int')
                
                return [generator[_type](p,i) for i in x]

            if type(x) is not int:
                raise TypeError('parameter k must be of type int')
            return generator[_type](p, x)

        except KeyError:
            raise ValueError("Invalid argument. Type is either 'first' or 'second'.")

    def cdf(self, x: Union[List[int], int], _type: str = 'first') -> Union[List[float], float]:
        """
        Args: 

            type(keyvalue ∈[fist, second]): defaults to first. Reconfigures the type of distribution.
            x (Union[List[int], int]): list of random variables

        Returns: 
            cumulative distribution evaluation to some point specified by k or scatter plot of geometric distribution.

        Note: there are two configurations of cdf. 
        """
        p = self.p

        try:
            generator = {'first': lambda p,k: 1-pow(1-p, k),    
                        'second': lambda p,k: 1-pow(1-p, k+1)}

            if isinstance(x, List):
                if any(type(i) is not int for i in x):
                    raise TypeError('parameter k must be of type int')
                return  [generator[_type](p,i) for i in x]

            if type(x) is not int:
                raise TypeError('parameter k must be of type int')
            return generator[_type](p, x)

        except KeyError:
            raise ValueError("Invalid argument. Type is either 'first' or 'second'.")



    def mean(self, _type='first') -> float:
        """
        Args:

            type(string): defaults to first type. Valid types: "first", "second".
        
        Returns: 
            the mean of Geometric Distribution.
        """

        if _type == "first":
            return 1 / self.p
        elif _type == "second":
            return (1 - self.p) / self.p
        else:  
            raise ValueError("Invalid argument. Type is either 'first' or 'second'.")

    def median(self, _type='first') -> int:
        """
        Args:

            type(string): defaults to first type. Valid types: "first", "second".
        
        Returns: 
            the median of Geometric Distribution.
        """
        if _type == "first":
            return _ceil(-1 / (_log2(1 - self.p)))
        elif _type == "second":
            return _ceil(-1 / (_log2(1 - self.p))) - 1
        else: 
            raise ValueError("Invalid argument. Type is either 'first' or 'second'.")

    def mode(self, _type: str = 'first') -> int:
        """
        Args:

            type(string): defaults to first type. Valid types: "first", "second".
        
        Returns: 
            the mode of Geometric Distribution.
        """
        if _type == "first":
            return 1
        elif _type == "second":
            return 0
        else: 
            raise ValueError("Invalid argument. Type is either 'first' or 'second'.")

    def var(self) -> float:
        """
        Returns: 
            the variance of Geometric Distribution.
        """
        return (1 - self.p) / self.p**2

    def skewness(self) -> float:
        """
        Returns: 
            the skewness of Geometric Distribution.
        """
        return (2 - self.p) / _sqrt(1 - self.p)

    def kurtosis(self) -> float:
        """
        Returns: 
            the kurtosis of Geometric Distribution.
        """
        return 6 + (self.p**2 / (1 - self.p))

    
    def keys(self) -> Dict[str, Union[float, int]]:
        """
        Summary statistic regarding the Geometric distribution which contains the following parts of the distribution:
        (mean, median, mode, var, std, skewness, kurtosis).

        Returns:
            Dict[str, Union[float, int]]
        """
        return {
            'mean': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.var(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }
