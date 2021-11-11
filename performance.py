"""
Evaluation of performance: native vs. numpy 
"""
from timeit import timeit as t
from typing import Dict
from tabulate import tabulate

test_size = [10, 100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000]
test_values = [9, 99, 999, 9_999, 99_999, 999_999, 9_999_999]

def predicate()->dict:
    result = {'list':[], 'numpy':[], 'map':[]}
    for ts,tv in zip(test_size, test_values):
        result['list'].append(t(stmt=f'any(i=={tv} for i in x)', 
                        setup=f'x = [*{range(ts)}]', number=50)/50)
        result['map'].append(t(stmt=f'any(map(lambda x: x=={tv}, x))', 
                        setup=f'x = [*{range(ts)}]', number=50)/50)
        result['numpy'].append(t(stmt=f'np.any(x=={tv})', 
                        setup=f'import numpy as np; x = np.arange({ts})', number=50)/50)
    return result

def vectorized()->dict:
    result = {'map':[], 'comprehension': [] , 'vectorized':[], 'return_list':[], 'piecewise':[]}
    for ts in test_size:
        result['map'].append(t(stmt=f'list(map(f, x))', 
            setup=f'f = lambda x: pow(x,2) if x%2==0 else x; x = [*{range(ts)}]', number=1))
        result['comprehension'].append(t(stmt=f'list(f(i) for i in x)', 
            setup=f'f = lambda x: pow(x,2) if x%2==0 else x; x = [*{range(ts)}]', number=1))
        result['vectorized'].append(t(stmt=f'np.vectorize(f)(x)', 
            setup=f'import numpy as np; f = lambda x: pow(x,2) if x%2==0 else x; x = np.arange({ts}, dtype=np.int64)', number=1))
        result['piecewise'].append(t(stmt=f'np.piecewise(x,[x%2==0, x%2!=0], [lambda x:x**2, lambda x: x])', 
            setup=f'import numpy as np; x = np.arange({ts}, dtype=np.int64)', number=1))
        result['return_list'].append(t(stmt=f'list(np.vectorize(f)(x))', 
            setup=f'import numpy as np; f = lambda x: pow(x,2) if x%2==0 else x; x = np.arange({ts}, dtype=np.int64)', number=1))
    return result


def array_conversion()->dict:
    result = {'fromiter':[], 'array':[]}
    for ts in test_size:
        result['fromiter'].append(t(stmt = 'np.fromiter(x, int)', 
            setup=f'import numpy as np; x = [*range({ts})]', number=1))
        result['array'].append(t(stmt='np.array(x)', 
            setup=f'import numpy as np; x = [*range({ts})]',number=1))

    return result

def output(d:Dict)->None:
    print(tabulate([[k,*v] for k,v in zip(d.keys(), d.values())], 
                    headers = ['size']+([str(i) for i in test_size]), tablefmt='github'))

output(array_conversion())

"""
| size   |         10 |        100 |        1000 |      10000 |      100000 |    1000000 |   10000000 |
|--------|------------|------------|-------------|------------|-------------|------------|------------|
| list   | 3.656e-06  | 2.922e-05  | 0.000289608 | 0.00144787 | 0.0121357   | 0.142511   |  1.51287   |
| numpy  | 1.7488e-05 | 4.4078e-05 | 3.1424e-05  | 1.8546e-05 | 0.000171592 | 0.00198056 |  0.0216845 |
| map    | 3.344e-06  | 3.121e-05  | 0.000289378 | 0.00167565 | 0.0180401   | 0.186457   |  2.41164   |

| size          |        10 |       100 |      1000 |     10000 |    100000 |   1000000 |   10000000 |
|---------------|-----------|-----------|-----------|-----------|-----------|-----------|------------|
| map           | 1.61e-05  | 0.0001095 | 0.0009683 | 0.0091869 | 0.114893  |  1.9749   |    8.4566  |
| comprehension | 4.78e-05  | 0.0001285 | 0.0011456 | 0.0068486 | 0.59592   |  2.7928   |    7.07799 |
| vectorized    | 0.0189447 | 0.0003916 | 0.0012698 | 0.0101151 | 0.460075  |  1.71298  |    7.42446 |
| return_list   | 0.0001673 | 0.0002965 | 0.0012804 | 0.0137902 | 0.728135  |  1.8353   |    8.22071 |
| piecewise     | 0.0108705 | 0.000225  | 0.0003739 | 0.0015247 | 0.0433989 |  0.194625 |    1.18457 |

| size     |       10 |      100 |      1000 |     10000 |    100000 |   1000000 |   10000000 |
|----------|----------|----------|-----------|-----------|-----------|-----------|------------|
| fromiter | 1.51e-05 | 2.63e-05 | 6.31e-05  | 0.0007202 | 0.0129518 | 0.0812345 |   0.571053 | same size
| array    | 1.84e-05 | 2.94e-05 | 0.0001979 | 0.0021895 | 0.0243982 | 0.212874  |   1.1908   | same size
"""