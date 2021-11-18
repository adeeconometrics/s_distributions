try:
    import numpy as np
    import scipy.special as ss
    from typing import Union, Tuple, Dict, List
    import math as m
    from univariate._base import Base
except Exception as e:
    print(f"some modules are missing {e}")


class SemiInfinite(Base):
    """
    Description:
        Base class for probability tags.
    """

    def __init__(self):
        if type(self) is SemiInfinite:
            raise TypeError('base class cannot be instantiated.')
