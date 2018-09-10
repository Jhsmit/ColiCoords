from functools import wraps
import colicoords
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


def box_mean(x_in, y_in, bins, storm_weight=False):
    """bins xvals in given bins using y_weight as weights"""
    i_sort = x_in.argsort()
    r_sorted = x_in[i_sort]
    y_in = y_in[i_sort] if y_in is not None else y_in
    bin_inds = np.digitize(r_sorted,
                           bins) - 1  # -1 to assure points between 0 and step are in bin 0 (the first)
    y_out = np.bincount(bin_inds, weights=y_in, minlength=len(bins))
    if y_in is not None and not storm_weight:
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


def pad_data(data, shape, mode='mean'):
    # todo doesnt work for 3d data
    pad_h = shape[1] - data.shape[1]
    assert pad_h >= 0
    pad_h_l = int(np.floor(pad_h/2))

    pad_v = shape[0] - data.shape[0]
    assert pad_v >= 0
    pad_v_t = int(np.floor(pad_v/2))
    d_out = colicoords.Data()
    for k, v in data.data_dict.items():

        if v.dclass == 'storm':
            v_out = v.copy()
            v_out['x'] += pad_h_l
            v_out['y'] += pad_v_t

            d_out.add_data(v_out, v.dclass, v.name)

        else:
            if mode == 'mean':
                f = np.concatenate([v[0, :], v[-1, :], v[1:-1, 0], v[1:-1, -1]])
                fill = f.mean()
            elif np.isscalar(mode):
                fill = mode
            else:
                raise ValueError('Invalid mode')

            v_out = (fill*np.ones(shape)).astype(v.dtype)
            v_out[pad_v_t:pad_v_t + data.shape[0], pad_h_l:pad_h_l + data.shape[1]] = v

            d_out.add_data(v_out, v.dclass, v.name)

    return d_out


def pad_cell(cell, shape, mode='mean'):
    pad_h = shape[1] - cell.data.shape[1]
    assert pad_h >= 0
    pad_h_l = int(np.floor(pad_h/2))

    pad_v = shape[0] - cell.data.shape[0]
    assert pad_v >= 0
    pad_v_t = int(np.floor(pad_v/2))

    d_out = pad_data(cell.data, shape, mode=mode)
    c_out = colicoords.Cell(d_out)
    c_out.name = cell.name

    c_out.coords.a0 = cell.coords.a0 - cell.coords.a1*pad_h_l + cell.coords.a2*pad_h_l**2 + pad_v_t
    c_out.coords.a1 = cell.coords.a1 - 2*cell.coords.a2*pad_h_l
    c_out.coords.a2 = cell.coords.a2  # redundant

    c_out.coords.xl = cell.coords.xl + pad_h_l
    c_out.coords.xr = cell.coords.xr + pad_h_l
    c_out.coords.r = cell.coords.r

    return c_out


def crop_data(data, shape):
    crop_h = data.shape[1] - shape[1]
    assert crop_h >= 0
    crop_h_l = int(np.floor(crop_h/2))

    crop_v = data.shape[0] - shape[0]
    assert crop_v >= 0
    crop_v_t = int(np.floor(crop_v/2))

    d_out = colicoords.Data()
    for k, v in data.data_dict.items():
        if v.dclass == 'storm':
            v_out = v.copy()
            v_out['x'] -= crop_h_l
            v_out['y'] -= crop_v_t

        elif v.ndim == 2:
            v_out = v.copy()[crop_v_t:-(crop_v-crop_v_t), crop_h_l:-(crop_h-crop_h_l)]
        elif v.ndim == 3:
            v_out = v.copy()[:, crop_v_t:-(crop_v-crop_v_t), crop_h_l:-(crop_h-crop_h_l)]

        d_out.add_data(v_out, v.dclass, v.name)

    return d_out


def crop_cell(cell, shape):
    #todo doesnt work when shape equal to current shape
    crop_h = cell.data.shape[1] - shape[1]
    assert crop_h >= 0
    crop_h_l = int(np.floor(crop_h/2))

    crop_v = cell.data.shape[0] - shape[0]
    assert crop_v >= 0
    crop_v_t = int(np.floor(crop_v/2))

    d_out = crop_data(cell.data, shape)
    c_out = colicoords.Cell(d_out)
    c_out.name = cell.name

    c_out.coords.a0 = cell.coords.a0 + cell.coords.a1*crop_h_l + cell.coords.a2*crop_h_l**2 - crop_v_t
    c_out.coords.a1 = cell.coords.a1 + 2*cell.coords.a2*crop_h_l
    c_out.coords.a2 = cell.coords.a2  # redundant

    c_out.coords.xl = cell.coords.xl - crop_h_l
    c_out.coords.xr = cell.coords.xr - crop_h_l
    c_out.coords.r = cell.coords.r

    return c_out