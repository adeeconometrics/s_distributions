try:
    from tabulate import tabulate
except Exception as e:
    print(f"some modules are missing {e}")

class Base:  
    def __init__(self):
        if type(self) is Base:
            raise TypeError('Discrete Univariate Base class cannot be instantiated.')

    def __str__(self)->str:
        pairs = self.summary()
        return tabulate([[k,v] for k,v in zip(pairs.keys(), pairs.values())],
                        tablefmt="github")

    def table(self)->None:
        """Prints out table summary. 
        """
        print(self)

    def rvs(self):  # (adaptive) rejection sampling implementation
        """
        returns random variate samples default NotImplemented
        """
        return "currently unsupported"

    def mean(self):
        """
        Default implementation of the mean.
        Returns NotImplemented.
        """
        return NotImplemented

    def median(self):
        """
        Default implementation of the median.
        Returns NotImplemented.
        """
        return NotImplemented

    def mode(self):
        """
        Default implementation of the mode.
        Returns NotImplemented.
        """
        return NotImplemented

    def var(self):
        """
        Default implementation of the variance.
        Returns NotImplemented.
        """
        return NotImplemented

    def std(self):
        """
        Default implementation of the standard deviation.
        Returns NotImplemented.
        """
        return NotImplemented

    def skewness(self):
        """
        Default implementation of skewness.
        Returns NotImplemented.
        """
        return NotImplemented

    def kurtosis(self):
        """
        Default implementation of kurtosis.
        Returns NotImplemented.
        """
        return NotImplemented

    def entropy(self):
        """
        Default implementation of entropy.
        Returns NotImplemented.
        """
        return NotImplemented

class Finite(Base):
    """
    Description:
        Base class for probability tags.
    """
    def __init__(self):
        if type(self) is Finite:
            raise TypeError('base class cannot be instantiated.')

class Infinite(Base):
    """
    Description:
        Base class for probability tags.
    """
    def __init__(self):
        if type(self) is Infinite:
            raise TypeError('base class cannot be instantiated.')