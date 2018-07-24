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


def box_mean(x_in, y_in, bins):
    """bins xvals in given bins using y_weight as weights"""
    i_sort = x_in.argsort()
    r_sorted = x_in[i_sort]
    y_in = y_in[i_sort] if y_in is not None else y_in
    bin_inds = np.digitize(r_sorted,
                           bins) - 1  # -1 to assure points between 0 and step are in bin 0 (the first)
    y_out = np.bincount(bin_inds, weights=y_in, minlength=len(bins))
    if y_in is not None:
        y_out /= np.bincount(bin_inds, minlength=len(bins))
    return np.nan_to_num(y_out)


#https://stackoverflow.com/questions/13728392/moving-average-or-running-mean
def running_mean(x_in, y_in, x_out, sigma=0.5):
    """Calculates average by sliding a gaussian kernel over `x_in`, `y_in` at points in `x_out`"""
    n_in = len(y_in)
    n_out = len(x_out)

    # Gaussian kernel
    x_in_mesh, x_out_mesh = np.meshgrid(x_in, x_out)
    gauss_kernel = np.exp(-np.square(x_in_mesh - x_out_mesh) / (2 * sigma**2))
    # Normalize kernel, such that the sum is one along axis 1
    normalization = np.tile(np.reshape(np.sum(gauss_kernel, axis=1), (n_out, 1)), (1, n_in))
    gauss_kernel_normalized = gauss_kernel / normalization
    # Perform running average as a linear operation
    y_out = gauss_kernel_normalized @ y_in

    return y_out


def gauss_2d(x, y, x_mu, y_mu, sigma):
    return np.exp( - (( (x - x_mu)**2 / (2*sigma**2) ) + ( (y - y_mu)**2 / (2*sigma**2) )) )