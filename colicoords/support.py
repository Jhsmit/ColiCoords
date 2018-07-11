from functools import wraps
import numpy as np


def allow_scalars(f):
    @wraps(f)
    def wrapper(self, *args, **kwargs):
        if np.all([np.isscalar(a) for a in args]):
            new_args = tuple(np.array([a]) for a in args)
            result = f(self, *new_args, **kwargs)
            try:
                return result.squeeze()
            except AttributeError:
                if type(result) == tuple:
                    return tuple(_res.squeeze() for _res in result)
                else:
                    return result
        else:
            return f(self, *args, **kwargs)
    return wrapper



def gauss_2d(x, y, x_mu, y_mu, sigma):
    return np.exp( - (( (x - x_mu)**2 / (2*sigma**2) ) + ( (y - y_mu)**2 / (2*sigma**2) )) )