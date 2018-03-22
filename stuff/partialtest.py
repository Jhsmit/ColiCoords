from functools import partial
from scipy.optimize import minimize
import numpy as np

kwargs = {'bar': 45}

def func(arg1, arg2, foo='bar', **kwargs):
    print(arg1, arg2, foo)
    for k, v in kwargs.items():
        print(k, v)

    return np.abs(arg1[0] - arg1[1])

g = partial(func, **kwargs)



par = [100, -500]


res = minimize(g, par, args=(1234,))

print(res.x)