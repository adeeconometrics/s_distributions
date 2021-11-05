try:
    import numpy as _np
    from numpy import euler_gamma as _euler_gamma
    from typing import Union, Tuple, Dict, List
    from math import sqrt as _sqrt, log as _log, pi as _pi, exp as _exp
    from univariate._base import SemiInfinite
except Exception as e:
    print(f"some modules are missing {e}")


class Rayleigh(SemiInfinite):
    """
    This class contains methods concerning Rayleigh Distirbution.
    Args:

        scale(float | x>0): scale
        randvar(float | x>=0): random variable

    Reference:
    - Wikipedia contributors. (2020, December 30). Rayleigh distribution. In Wikipedia, The Free Encyclopedia.
    Retrieved 09:37, January 8, 2021, from https://en.wikipedia.org/w/index.php?title=Rayleigh_distribution&oldid=997166230

    - Weisstein, Eric W. "Rayleigh Distribution." From MathWorld--A Wolfram Web Resource.
    https://mathworld.wolfram.com/RayleighDistribution.html
    """

    def __init__(self, scale: float, randvar: float):
        if randvar < 0:
            raise ValueError(
                'random variable should be a positive number. Entered value: {}'.format(randvar))
        if scale < 0:
            raise ValueError('scale parameter should be a positive number.')

        self.scale = scale
        self.randvar = randvar

    def pdf(self, x: Union[List[float], _np.ndarray] = None) -> Union[float, _np.ndarray]:
        """
        Args:

            x (List[float], numpy.ndarray): random variable or list of random variables

        Returns:
            either probability density evaluation for some point or plot of Raylegh distribution.
        """
        sig = self.scale  # scale to sig
        randvar = self.randvar

        if x is not None:
            if not isinstance(x, (_np.ndarray, List)):
                raise TypeError(
                    f'parameter x only accepts List types or numpy.ndarray')
            else:
                x = _np.array(x)
                return x/pow(sig, 2) * _np.exp(_np.power(-x, 2)/(2*pow(sig, 2)))

        return randvar/pow(sig, 2) * _exp(pow(-randvar, 2)/(2*pow(sig, 2)))

    def cdf(self, x: Union[List[float], _np.ndarray] = None) -> Union[float, _np.ndarray]:
        """
        Args:

            x (List[float], numpy.ndarray): random variable or list of random variables

        Returns:
            either cumulative distribution evaluation for some point or plot of Raylegh distribution.
        """
        sig = self.scale
        randvar = self.randvar

        if x is not None:
            if not isinstance(x, (_np.ndarray, List)):
                raise TypeError(
                    f'parameter x only accepts List types or numpy.ndarray')
            else:
                x = _np.array(x)
                return 1-_np.exp(-_np.power(x, 2)/(2*sig**2))

        return 1-_exp(-randvar**2/(2*sig**2))

    def mean(self) -> float:
        """
        Returns: Mean of the Rayleigh distribution.
        """
        return self.scale*_sqrt(_pi/2)

    def median(self) -> float:
        """
        Returns: Median of the Rayleigh distribution.
        """
        return self.scale*_sqrt(2*_log(2))

    def mode(self) -> float:
        """
        Returns: Mode of the Rayleigh distribution.
        """
        return self.scale

    def var(self) -> float:
        """
        Returns: Variance of the Rayleigh distribution.
        """
        return (4-_pi)/2*pow(self.scale, 2)

    def std(self) -> float:
        """
        Returns: Standard deviation of the Rayleigh distribution
        """
        return _sqrt((4-_pi)/2*pow(self.scale, 2))

    def skewness(self) -> float:
        """
        Returns: Skewness of the Rayleigh distribution.
        """
        return (2*_sqrt(_pi)*(_pi-3))/pow((4-_pi), 3/2)

    def kurtosis(self) -> float:
        """
        Returns: Kurtosis of the Rayleigh distribution.
        """
        return -(6*pow(_pi, 2)-24*_pi+16)/pow(4-_pi, *2)

    def entropy(self) -> float:
        """
        Returns: differential entropy of the Rayleigh distribution.

        Reference: Park, S.Y. & Bera, A.K.(2009). Maximum entropy autoregressive conditional heteroskedasticity model. Elsivier.
        link: http://wise.xmu.edu.cn/uploadfiles/paper-masterdownload/2009519932327055475115776.pdf
        """
        return 1+_log(self.scale/_sqrt(2))+(_euler_gamma/2)

    def summary(self, display=False) -> Union[None, Tuple[str, str, str, str, str, str, str]]:
        """
        Returns:  summary statistic regarding the Rayleigh distribution which contains the following parts of the distribution:
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
        Summary statistic regarding the Rayleigh distribution which contains the following parts of the distribution:
        (mean, median, mode, var, std, skewness, kurtosis).

        Returns:
            Dict[str, Union[float, Tuple[float]]]: [description]
        """
        return {
            'mean': self.mean(), 'median': self.median(), 'mode': self.mode(),
            'var': self.var(), 'std': self.std(), 'skewness': self.skewness(), 'kurtosis': self.kurtosis()
        }