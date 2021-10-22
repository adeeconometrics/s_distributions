try:
    import numpy as _np
    from typing import Union, Tuple, Dict, List
    from math import sqrt as _sqrt, log as _log
    from univariate._base import BoundedInterval
except Exception as e:
    print(f"some modules are missing {e}")


class Bernoulli(BoundedInterval):
    """
    This class contains methods concerning Continuous Bernoulli Distirbution.
    The continuous Bernoulli distribution arises in deep learning and computer vision,
    specifically in the context of variational autoencoders, for modeling the
    pixel intensities of natural images

    Args:

        shape(float): parameter
        randvar(float | x in [0,1]): random variable

    Methods:

        - pdf for probability density function.
        - cdf for cumulative distribution function.
        - pvalue for p-values.
        - mean for evaluating the mean of the distribution.
        - median for evaluating the median of the distribution.
        - mode for evaluating the mode of the distribution.
        - var for evaluating the variance of the distribution.
        - std for evaluating the standard deviation of the distribution.
        - skewness for evaluating the skewness of the distribution.
        - kurtosis for evaluating the kurtosis of the distribution.
        - entropy for differential entropy of the distribution.
        - summary for printing the summary statistics of the distribution.
        - keys for returning a dictionary of summary statistics.

    Reference:
    - Wikipedia contributors. (2020, November 2). Continuous Bernoulli distribution. In Wikipedia, The Free Encyclopedia.
    Retrieved 02:37, January 14, 2021, from https://en.wikipedia.org/w/index.php?title=Continuous_Bernoulli_distribution&oldid=986761458
    - Kingma, D. P., & Welling, M. (2013). Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114.
    - Kingma, D. P., & Welling, M. (2014, April). Stochastic gradient VB and the variational auto-encoder.
    In Second International Conference on Learning Representations, ICLR (Vol. 19).
    - Ganem, G & Cunningham, J.P. (2019). The continouous Bernoulli: fixing a pervasive error in variational autoencoders. https://arxiv.org/pdf/1907.06845.pdf
    """

    def __init__(self, shape: float, randvar: float):
        if randvar < 0 or randvar > 1:
            raise ValueError(
                'random variable should only be in between 0 and 1. Entered value: {}'.format(randvar))
        if shape < 0 or shape > 1:
            raise ValueError(
                'shape parameter a should only be in between 0 and 1. Entered value:{}'.format(shape))

        self.shape = shape
        self.randvar = randvar

    def pdf(self, x: Union[List[float], _np.ndarray] = None) -> Union[float, _np.ndarray]:
        """
        Args:

            x (List[float], numpy.ndarray): random variable or list of random variables

        Returns:
            either probability density evaluation for some point or plot of Continuous Bernoulli distribution.
        """
        def __C(shape: float):
            return (2*_np.arctanh(1-2*shape)) / (1-2*shape) if shape != 0.5 else 2

        if x is not None:
            if not isinstance(x, (_np.ndarray, List)):
                raise TypeError(
                    f'parameter x only accepts List types or numpy.ndarray')
            else:
                x = _np.array(x)
                return __C(self.shape) * _np.power(shape, x)*_np.power(1-shape, 1-x)

        return __C(self.shape)*pow(shape, self.randvar)*pow(1-shape, 1 - self.randvar)

    def cdf(self, x: Union[List[float], _np.ndarray] = None) -> Union[float, _np.ndarray]:
        """
        Args:

            x (List[float], numpy.ndarray): random variable or list of random variables

        Returns:
            either cumulative distribution evaluation for some point or plot of Continuous Bernoulli distribution.
        """
        shape = self.shape
        randvar = self.randvar

        if x is not None:
            if not isinstance(x, (_np.ndarray, List)):
                raise TypeError(
                    f'parameter x only accepts List types or numpy.ndarray')
            else:
                x = _np.array(x)
                return (_np.power(shape, x)*_np.power(1-shape, 1-x) + shape - 1)/(1-2*shape) if shape != 0.5 else x

        return (shape**x*pow(1-shape, 1-x)+shape-1)/(2*shape-1) if shape != 0.5 else x

    def pvalue(self, x_lower=0, x_upper=None) -> Optional[float]:
        """
        Args:

            x_lower(float): defaults to 0. Defines the lower value of the distribution. Optional.
            x_upper(float): defaults to None. If not defined defaults to random variable x. Optional.

            Note: definition of x_lower and x_upper are only relevant when probability is between two random variables.
            Otherwise, the default random variable is x.

        Returns:
            p-value of the Continuous Bernoulli distribution evaluated at some random variable.
        """
        if x_upper == None:
            x_upper = self.randvar
        if x_lower > x_upper:
            raise Exception(
                'lower bound should be less than upper bound. Entered values: x_lower:{} x_upper:{}'.format(x_lower, x_upper))

        def __cdf(shape, x):
            if shape != 0.5:
                return (shape**x*pow(1-shape, 1-x)+shape-1)/(2*shape-1)
            else:
                x
        return __cdf(self.shape, x_upper)-__cdf(self.shape, x_lower)

    def mean(self) -> float:
        """
        Returns: Mean of the Continuous Bernoulli distribution.
        """
        shape = self.shape
        if shape == 0.5:
            return 0.5
        return shape/(2*shape-1)+(1/(2*_np.arctanh(1-2*shape)))

    def var(self) -> float:
        """
        Returns: Variance of the Continuous Bernoulli distribution.
        """
        shape = self.shape
        if shape == 0.5:
            return 1/12
        return shape/((2*shape-1)**2)+1/(2*_np.arctanh(1-2*shape))**2

    def std(self) -> float:
        """
        Returns: Standard deviation of the Continuous Bernoulli distribution
        """
        return _sqrt(self.var())

    def summary(self, display=False) -> Union[None, Tuple[str, str, str, str, str, str, str]]:
        """
        Returns:  summary statistic regarding the Bernoulli distribution which contains the following parts of the distribution:
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

    def keys(self) -> Dict[str, Union[float, Tuple[float]]]:
        """
        Summary statistic regarding the Bernoulli distribution which contains the following parts of the distribution:
        (mean, median, mode, var, std, skewness, kurtosis).

        Returns:
            Dict[str, Union[float, Tuple[float]]]: [description]
        """
        return {
            'mean': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.var(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }
