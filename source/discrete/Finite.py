try:
    import numpy as np
    import scipy.special as ss
    from discrete._base import Base
    from typing import Union, Tuple, Dict, List
    import math as m
except Exception as e:
    print(f'some modules are missin{e}')


class Finite(Base):
    """
    Description:
        Base class for probability tags.
    """

    def __init__(self):
        if type(self) is Finite:
            raise TypeError('base class cannot be instantiated.')


class Bernoulli(Finite):
    """
    This class contains methods concerning the Bernoulli Distribution. Bernoulli Distirbution is a special
    case of Binomial Distirbution [#]_ [#]_. 

    .. math:: 
        \\text{Bernoulli} (x;p) = p^n (1-p)^{1-x}

    Args:

        - p (float) : event of success. Either 0 or 1. 
        - x (int) : possible outcomes. Either 0 or 1.

    References:
        .. [#] Weisstein, Eric W. "Bernoulli Distribution." From MathWorld--A Wolfram Web Resource. https://mathworld.wolfram.com/BernoulliDistribution.html

        .. [#] Wikipedia contributors. (2020, December 26). Bernoulli distribution. https://en.wikipedia.org/w/index.php?title=Bernoulli_distribution&oldid=996380822
    """

    def __init__(self, p: float):
        if p < 0 or p > 1:
            raise ValueError('parameter k is constrained in ∈ [0,1]')
        self.p = p

    def pmf(self, x: Union[List[int], int, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Args:
            x (Union[List[int], int, numpy.ndarray]): random variable(s)

        Raises:
            ValueError: when there exist a value of x that is not 0 or 10

        Returns:
            Union[float, numpy.ndarray]: evaluation of pmf at x
        """
        p = self.p

        if isinstance(x, (List, np.ndarray)):
            if not type(x) is np.array:
                x = np.array(x)
            if np.all(np.logical_or(x == 0, x == 1)) == False:
                raise ValueError('all x must either be 1 or 0')
            return np.piecewise(x, [x == 0, x != 0], [1-p, p])

        if x != 1 or x != 0:
            raise ValueError('all x must either be 1 or 0')
        return 1-p if x == 0 else p

    @staticmethod
    def pmf_s(p: float, x: Union[List[int], int, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Args:
            p (float): event of success, either 0 or 1
            x (Union[List[int], int, numpy.ndarray]): random variable(s)

        Raises:
            ValueError: when parameter p does not belong to the domain [0,1]
            ValueError: when there exist a value in a random variable that is not 0 or 1

        Returns:
            Union[float, numpy.ndarray]: evaluation of pmf at x
        """
        if p < 0 or p > 1:
            raise ValueError('parameter p is constrained in ∈ [0,1]')

        if isinstance(x, (List, np.ndarray)):
            if not type(x) is np.array:
                x - np.array(x)
            if np.all(np.logical_or(x == 0, x == 1)) == False:
                raise ValueError('all x must either be 1 or 0')
            return np.piecewise(x, [x == 0, x != 0], [1-p, p])

        if x != 1 or x != 0:
            raise ValueError('all x must either be 1 or 0')
        return 1-p if x == 0 else p

    def cdf(self, x: Union[List[int], int, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Args:
            x (Union[List[int], int, numpy.ndarray]): data point(s) of interest

        Raises:
            ValueError: when there exist a value of x not equal to 0 or 1

        Returns:
            Union[float, numpy.ndarray]: evaluation of cdf at x
        """
        p = self.p

        if isinstance(x, (List, np.ndarray)):
            if not type(x) is np.array:
                x = np.array(x)
            if np.any(np.logical_or(x != 0, x != 1)):
                raise ValueError('all x must either be 1 or 0')
            return np.piecewise(x, [x < 0, (x >= 0)*(x < 1), x >= 1], [0.0, 1-p, 1.0])

        if x != 1 or x != 0:
            raise ValueError('all x must either be 1 or 0')
        return 0.0 if x < 0 else (1-p if x >= 0 and x > 1 else 1)

    def mean(self) -> float:
        """
        Returns:
            float: mean of Bernoulli distribution
        """
        return self.p

    def median(self) -> Union[List[int], int]:
        """
        Returns:
            Union[List[int], int]: median of Bernoulli distribution
        """
        p = self.p
        if p < 0.5:
            return 0
        if p == 0.5:
            return [0, 1]
        return 1

    def mode(self) -> Union[Tuple[int, int], int]:
        """
        Returns:
            Union[Tuple[int, int], int]: mode of Bernoulli distribution 
        """
        p = self.p
        if p < 0.5:
            return 0
        if p == 0.5:
            return (0, 1)
        return 1

    def var(self) -> float:
        """
        Returns:
            float: variance of Bernoulli distribution
        """
        p = self.p
        q = 1 - p
        return p * q

    def std(self) -> float:
        """
        Returns:
            float: standard deviation of Bernoulli distribution
        """
        p = self.p
        q = 1 - p
        return m.sqrt(p * q)

    def skewness(self) -> float:
        """
        Returns:
            float: skewness of Bernoulli distribution
        """
        p = self.p
        q = 1 - p
        return (q - p) / m.sqrt(p * q)

    def kurtosis(self) -> float:
        """ 
        Returns:
            float: kurtosis of Bernoulli distribution
        """
        p = self.p
        q = 1 - p
        return (1 - 6 * p * q) / (p * q)

    def summary(self) -> Dict[str, Union[int, float, List[int], Tuple[int, int]]]:
        """
        Returns:
            Dictionary of Bernoulli distirbution moments. This includes standard deviation. 
        """
        return {
            'mean': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.var(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }


class Binomial(Finite):
    """
    This class contains functions for finding the probability mass function and 
    cumulative distribution function for binomial distirbution [#]_ [#]_ [#]_. 

    .. math::
        \\text{Binomial}(x;n,p) = \\binom{n}{x} p^k (1-p)^{n-x}

    Args:

        n (int): number  of trials
        p (float): success probability for each trial. Where 0 <= p <= 1.
        x (int): number of successes 


    References:
        .. [#] NIST/SEMATECH e-Handbook of Statistical Methods (2012). Binomial Distribution. Retrieved at http://www.itl.nist.gov/div898/handbook/, December 26, 2000.
        .. [#] Wikipedia contributors. (2020, December 19). Binomial distribution. In Wikipedia, The Free Encyclopedia. Retrieved 07:24, December 26, 2020, from https://en.wikipedia.org/w/index.php?title=Binomial_distribution&oldid=995095096
        .. [#] Weisstein, Eric W. "Binomial Distribution." From MathWorld--A Wolfram Web Resource. https://mathworld.wolfram.com/BinomialDistribution.html
    """

    def __init__(self, n: int, p: float):
        if p < 0 or p > 1:
            raise ValueError('parameter p is constrained to ∈ [0,1]')

        self.n = n
        self.p = p

    def pmf(self, x: Union[List[int], int, np.ndarray]) -> Union[int, np.ndarray]:
        """
        Args:
            x (Union[List[int], int]): random variable or list of random variables

        Returns:
            Union[int, numpy.ndarray]: evaluation of pmf at x
        """
        n = self.n
        p = self.p

        if isinstance(x, (List, np.ndarray)):
            if not type(x) is np.ndarray:
                x = np.array(x)
        return ss.binom(n, x)*p**x*(1-p)**(n-x)

    def cdf(self, x: Union[int, List[int], np.ndarray]) -> Union[int, np.ndarray]:
        """
        Args:
            x (Union[int, List[int], np.ndarray]): random variable or list of random variables

        Returns:
            Union[int, numpy.ndarray]: evaluation of cdf at x
        """

        n = self.n
        p = self.p

        if isinstance(x, List):
            if not type(x) is np.ndarray:
                x = np.array(x)
        return ss.betainc(n-x, 1+x, 1-p)

    def mean(self) -> float:
        """
        Returns: 
            the mean of Binomial Distribution.
        """
        return self.n * self.p

    def median(self) -> Tuple[int, int]:
        """
        Returns: 
            the median of Binomial Distribution. Either one defined in the tuple of result.
        """
        n = self.n
        p = self.p
        return m.floor(n * p), m.ceil(n * p)

    def mode(self) -> Tuple[int, int]:
        """
        Returns: 
            the mode of Binomial Distribution. Either one defined in the tuple of result.
        """
        n = self.n
        p = self.p
        return m.floor((n + 1) * p), m.ceil((n + 1) * p) - 1

    def var(self) -> float:
        """
        Returns: 
            the variance of Binomial Distribution.
        """
        n = self.n
        p = self.p
        q = 1 - p
        return n * p * q

    def skewness(self) -> float:
        """
        Returns: 
            the skewness of Binomial Distribution.
        """
        n = self.n
        p = self.p
        q = 1 - p
        return (q - p) / m.sqrt(n * p * q)

    def kurtosis(self) -> float:
        """
        Returns: 
            the kurtosis of Binomial Distribution.
        """
        n = self.n
        p = self.p
        q = 1 - p
        return (1 - 6 * p * q) / (n * p * q)

    def keys(self) -> Dict[str, Union[float, int, Tuple[int, int]]]:
        """
        Returns:
            Dictionary of Binomial distirbution moments. This includes standard deviation. 
        """
        return {
            'mean': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.var(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }


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

    def __init__(self, p: float):
        if p < 0 or p > 1:
            raise ValueError('parameter p is constrained at')

        self.p = p

    def pmf(self, x: Union[List[int], int, np.ndarray], _type: str = 'first') -> Union[np.ndarray, float]:
        """
        Args:
            x (Union[List[int], int, numpy.ndarray]): random variable(s)
            _type (str, optional): optional specifier for modifying the type of Geometric distribution. Defaults to 'first'.

        Raises:
            TypeError: when random variable(s) are not of type int
            ValueError: when a _type parameter is not 'first' or second

        Returns:
            Union[numpy.ndarray, float]: evaluation of pmf at x
        """
        p = self.p
        try:
            generator = {'first': lambda p, k: pow(1-p, k-1)*p,
                         'second': lambda p, k: pow(1-p, k)*p}

            if isinstance(x, (List, np.ndarray)):
                if not type(x) is np.ndarray:
                    x = np.array(x)
                if not np.issubdtype(x[0], np.integer):
                    raise TypeError('parameter k must be of type int')
                return np.vectorize(generator[_type])(p, x)

            if type(x) is not int:
                raise TypeError('parameter k must be of type int')
            return generator[_type](p, x)

        except KeyError:
            raise ValueError(
                "Invalid argument. Type is either 'first' or 'second'.")

    def cdf(self, x: Union[List[int], int], _type: str = 'first') -> Union[np.ndarray, float]:
        """
        Args:
            x (Union[List[int], int, numpy.ndarray]): random variable(s)
            _type (str, optional): optional specifier for modifying the type of Geometric distribution. Defaults to 'first'.

        Raises:
            TypeError: when random variable(s) are not of type int
            ValueError: when a _type parameter is not 'first' or second

        Returns:
            Union[numpy.ndarray, float]: evaluation of cdf at x
        """
        p = self.p

        try:
            generator = {'first': lambda p, k: 1-pow(1-p, k),
                         'second': lambda p, k: 1-pow(1-p, k+1)}

            if isinstance(x, (List, np.ndarray)):
                if not type(x) is np.ndarray:
                    x = np.array(x)
                if not np.issubdtype(x[0], np.integer):
                    raise TypeError('parameter k must be of type int')
                return np.vectorize(generator[_type])(p, x)

            if type(x) is not int:
                raise TypeError('parameter k must be of type int')
            return generator[_type](p, x)

        except KeyError:
            raise ValueError(
                "Invalid argument. Type is either 'first' or 'second'.")

    def mean(self, _type='first') -> float:
        """
        Args:
            _type (str, optional): modifies the type of Geometric distribution. Defaults to 'first'.

        Raises:
            ValueError: when _type is not 'first' or 'second'

        Returns:
            float: mean of Geometric distribution
        """

        if _type == "first":
            return 1 / self.p
        elif _type == "second":
            return (1 - self.p) / self.p
        else:
            raise ValueError(
                "Invalid argument. Type is either 'first' or 'second'.")

    def median(self, _type='first') -> int:
        """
        Args:
            _type (str, optional): modifies the type of Geometric distribution. Defaults to 'first'.

        Raises:
            ValueError: when _type is not 'first' or 'second'

        Returns:
            int: median of Geometric distribution
        """
        if _type == "first":
            return m.ceil(-1 / (m.log2(1 - self.p)))
        elif _type == "second":
            return m.ceil(-1 / (m.log2(1 - self.p))) - 1
        else:
            raise ValueError(
                "Invalid argument. Type is either 'first' or 'second'.")

    def mode(self, _type: str = 'first') -> int:
        """
        Args:
            _type (str, optional): modifies the type of Geometric distribution. Defaults to 'first'.

        Raises:
            ValueError: when _type is not 'first' or 'second'

        Returns:
            int: mode of Geometric distribution
        """
        if _type == "first":
            return 1
        elif _type == "second":
            return 0
        else:
            raise ValueError(
                "Invalid argument. Type is either 'first' or 'second'.")

    def var(self) -> float:
        """
        Returns:
            float: variance of Geometric distribution
        """
        return (1 - self.p) / self.p**2

    def skewness(self) -> float:
        """
        Returns:
            float: skewness of Geometric distribution
        """
        return (2 - self.p) / m.sqrt(1 - self.p)

    def kurtosis(self) -> float:
        """
        Returns:
            float: kurtosis of Geometric distribution
        """
        return 6 + (self.p**2 / (1 - self.p))

    def keys(self) -> Dict[str, Union[float, int]]:
        """
        Returns:
            Dictionary of Geometric distirbution moments. This includes standard deviation. 
        """
        return {
            'mean': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.var(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }


class Hypergeometric(Finite):
    """
    This class contains methods concerning pmf and cdf evaluation of the hypergeometric distribution. 
    Describes the probability if k successes (random draws for which the objsect drawn has specified deature)
    in n draws, without replacement, from a finite population size N that contains exactly K objects with that
    feature, wherein each draw is either a success or a failure [#]_ [#]_. 

    .. math:: \\text{Hypergeometric}(N,K,k,n) = {{{K \\choose k}{{N-K} \\choose {n-k}}} \\over {N \\choose n}}

    Args:

        N(int): population size
        K(int): number of success states in the population
        k(int): number of observed successes
        n(int): number of draws 

    References:
        .. [#] Weisstein, Eric W. "Hypergeometric Distribution." From MathWorld--A Wolfram Web Resource. https://mathworld.wolfram.com/HypergeometricDistribution.html
        .. [#] Wikipedia contributors. (2020, December 22). Hypergeometric distribution. In Wikipedia, The Free Encyclopedia. Retrieved 08:38, December 26, 2020, from https://en.wikipedia.org/w/index.php?title=Hypergeometric_distribution&oldid=995715954

    """

    def __init__(self, N: int, K: int, k: int, n: int):
        if type(N) and type(n) and type(K) and type(k) is not int:
            raise TypeError('all parameters must be of type int')

        if any(i < 0 for i in [N, K, k, n]):
            raise ValueError('parameters must be positive integer')

        self.N = N
        self.K = K
        self.k = k
        self.n = n

    def pmf(self) -> float:
        """
        Returns:
            float: evaluation of pmf
        """
        n = self.n
        k = self.k
        N = self.N
        K = self.K

        # assumes n>k
        return ss.binom(K, k)*ss.binom(N-K, n-k)/ss.binom(N, n)

    def cdf(self):
        """
        Args:

            x (List[int]): random variable or list of random variables

        Returns: 
            either cumulative density evaluation for some point or scatter plot of Hypergeometric distribution.
        """
        return NotImplemented

    def mean(self) -> float:
        """
        Returns: 
            the mean of Hypergeometric Distribution.
        """
        return self.n * (self.K / self.N)

    def median(self) -> str:
        """
        Returns: 
            the median of Hypergeometric Distribution. Currently unsupported or undefined.
        """
        return "undefined"

    def mode(self) -> Tuple[int, int]:
        """
        Returns:
            Tuple[int, int]: mode
        """
        n = self.n
        N = self.N
        K = self.K
        return m.ceil(((n + 1) * (K + 1)) / (N + 2)) - 1, m.floor(
            ((n + 1) * (K + 1)) / (N + 2))

    def var(self) -> float:
        """
        Returns:
            float: variance
        """
        n = self.n
        N = self.N
        K = self.K
        return n * (K / N) * ((N - K) / N) * ((N - n) / (N - 1))

    def skewness(self) -> float:
        """
        Returns:
            float: skewness
        """
        n = self.n
        N = self.N
        K = self.K
        return ((N - 2 * K) * pow(N - 1, 1 / 2) *
                (N - 2 * n)) / (m.sqrt(n * K * (N - K) * (N - n)) * (N - 2))

    def kurtosis(self) -> float:
        """
        Returns:
            float: kurtosis
        """
        n = self.n
        N = self.N
        K = self.K
        scale = 1 / (n * K*(N - K) * (N - n) * (N - 2) * (N - 3))
        return scale * ((N - 1) * N**2 * (N * (N + 1) - (6 * K * (N - K)) -
                                          (6 * n * (N - n))) +
                        (6 * n * K*(N - K) * (N - n) * (5 * N - 6)))

    def summary(self) -> Dict[str, Union[float, str, Tuple[int, int]]]:
        """
        Returns:
            Dictionary of Hypergeometric distirbution moments. This includes standard deviation. 
        """
        return {
            'mean': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.var(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }


class Uniform(Finite):
    """
    This contains methods for finding the probability mass function and 
    cumulative distribution function of Uniform distribution. Incudes scatter plot [#]_. 

    .. math:: \\text{Uniform} (a,b) = {\\begin{cases}{\\frac {1}{b-a}}&\\mathrm {for} \\ a\\leq x\\leq b,\\ \\[8pt]0&\\mathrm {for} \\ x<a\ \\mathrm {or} \\ x>b\\end{cases}}

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

    def pmf(self, x: Union[List[int], np.ndarray, int]) -> Union[float,  np.ndarray]:
        """
        Args:
            x (Union[List[int], np.ndarray, int]): random variable(s)

        Returns:
            Union[float,  np.ndarray]: evaluation of pmf at x
        """

        if isinstance(x, (List, np.ndarray)):
            x = np.empty(len(x))
            x[:] = 1/self.n
            return x
        return 1 / self.n

    def cdf(self, x: Union[List[int], np.ndarray, int]) -> Union[float,  np.ndarray]:
        """
        Args:
            x (Union[List[int], np.ndarray, int]): data point(s)

        Returns:
            Union[float,  np.ndarray]: evaluation of cdf at x
        """

        a, b, n = self.a, self.b, self.n

        if isinstance(x, (List, np.ndarray)):
            if not type(x) is np.ndarray:
                x = np.array(x)
            if not np.issubdtype(x[0], np.integer):
                raise TypeError('random variables must be of type integer')
            return np.piecewise(x, [x < a, (x >= a) & (x <= b), x > b], [0.0, lambda x: (np.floor(x-a) + 1)/n, 1.0])
        return (m.floor(x-a) + 1)/n if x >= a and x <= b else (0.0 if x < a else 1.0)

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


# class Zipf(Finite):
#     ...
