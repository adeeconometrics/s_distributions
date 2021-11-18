try:
    import numpy as np
    import scipy.special as ss
    from typing import Union, Tuple, Dict, List
    import math as m
    from univariate._base import BoundedInterval
except Exception as e:
    print(f"some modules are missing {e}")

