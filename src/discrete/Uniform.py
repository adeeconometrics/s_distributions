try:
    import numpy as _np
    from typing import Union, Tuple, Dict, List
    from discrete._base import Base
except Exception as e:
    print(f"some modules are missing {e}")


class Uniform(Base):
    """
    This contains methods for finding the probability mass function and 
    cumulative distribution function of Uniform distribution. Incudes scatter plot. 

    Args: 

        data (int): sample size

    Methods

        - pdf for evaluating or plotting probability mass function
        - cdf for evaluating or plotting cumulative distribution function
        - mean for evaluating the mean of the distribution.
        - median for evaluating the median of the distribution.
        - mode for evaluating the mode of the distribution.
        - var for evaluating the variance of the distribution.
        - skewness for evaluating the skewness of the distribution.
        - kurtosis for evaluating the kurtosis of the distribution.
        - summary for printing the summary statistics of the distribution.
        - keys for returning a dictionary of summary statistics.

    Reference:
    - NIST/SEMATECH e-Handbook of Statistical Methods (2012). Uniform Distribution. Retrieved from http://www.itl.nist.gov/div898/handbook/, December 26, 2020.
    """

    def __init__(self, a: int, b: int):
        if type(a) and type(b) is not int:
            raise TypeError(f'parameter a and b should be of type integer')

        self.a = a
        self.b = b
        self.n = abs(b-a+1)

    def pmf(self, x: List[float] = None) -> Union[float, List[float]]:
        """
        Args:

            x (List[int]): random variable or list of random variables

        Returns: 
            either probability mass evaluation for some point or scatter plot of Uniform distribution.
        """
        if x is not None and issubclass(x, List):
            return [1/self.n]*len(x)
        return 1 / self.n

    def cdf(self, x: Union[List[float], int]) -> Union[float, List[float]]:
        """
        Args:

            x (List[int]): random variable or list of random variables

        Returns: 
            either cumulative density evaluation for some point or scatter plot of Unifom distribution.
        """

        if issubclass(x, List):
            return [i/self.n for i in x]
        return x/self.n

    def mean(self) -> float:
        """
        Returns the mean of Uniform Distribution.
        """
        return (self.a + self.b) / 2

    def median(self) -> float:
        """
        Returns the median of Uniform Distribution.
        """
        return (self.a + self.b) / 2

    def mode(self) -> Tuple[int, int]:
        """
        Returns the mode of Uniform Distribution.
        """
        return (self.a, self.b)

    def var(self) -> float:
        """
        Returns the variance of Uniform Distribution.
        """
        return (self.b - self.a)**2 / 12

    def skewness(self) -> int:
        """
        Returns the skewness of Uniform Distribution.
        """
        return 0

    def kurtosis(self) -> float:
        """
        Returns the kurtosis of Uniform Distribution.
        """
        return -6 / 5

    def summary(self, display=False) -> Union[None, Tuple[str, str, str, str, str, str, str]]:
        """
        Returns:  summary statistic regarding the Uniform-distribution which contains the following parts of the distribution:
                (mean, median, mode, var, std, skewness, kurtosis). If the display parameter is True, the function returns None
                and prints out the summary of the distribution. 
        """
        if display == True:
            cstr = " summary statistics "
            print(cstr.center(40, "="))
            print(f"mean: {self.mean()}", f"median: {self.median()}",
                  f"mode: {self.mode()}", f"var: {self.var()}", f"std: {self.std()}",
                  f"skewness: {self.skewness()}", f"kurtosis: {self.kurtosis()}", sep='\n')

            return None
        else:
            return (f"mean: {self.mean()}", f"median: {self.median()}",
                    f"mode: {self.mode()}", f"var: {self.var()}", f"std: {self.std()}",
                    f"skewness: {self.skewness()}", f"kurtosis: {self.kurtosis()}")

    def keys(self) -> Dict[str, Union[float, int]]:
        """
        Summary statistic regarding the Uniform-distribution which contains the following parts of the distribution:
        (mean, median, mode, var, std, skewness, kurtosis).

        Returns:
            Dict[str, Union[float, int]]: [description]
        """
        return {
            'mean': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.var(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }
