from functools import wraps
import numpy as np


def allow_scalars(f):
    @wraps(f)
    def wrapper(self, *args):
        if np.all([np.isscalar(a) for a in args]):
            new_args = tuple(np.array([a]) for a in args)
            return f(self, *new_args).squeeze()
        else:
            return f(self, *args)
    return wrapper
