"""
Evaluation of performance: native vs. numpy 
"""
from timeit import timeit as t
from tabulate import tabulate

test_size = [100, 1_000, 10_000, 100_000]
test_values = [99, 999, 9_999, 99_999]

def predicate(size:list, value:list)->dict:
    result = {'list':[], 'numpy':[]}
    for ts,tv in zip(test_size, test_values):
        result['list'].append(t(stmt=f'any(i=={tv} for i in x)', 
                        setup=f'f = lambda x: pow(x,2) if x%2==0 else x; x = [*{range(ts)}]', number=100))
        result['numpy'].append(t(stmt=f'np.any(x=={tv})', 
                        setup=f'import numpy as np; f = lambda x: pow(x,2) if x%2==0 else x; x = np.arange({ts})', number=100))
    return result

p = predicate(test_size, test_values)
print(tabulate([[k,*v] for k,v in zip(p.keys(), p.values())], 
                headers = ['size']+([str(i) for i in test_size]), tablefmt='github'))