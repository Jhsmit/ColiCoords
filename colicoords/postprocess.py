import numpy as np
from colicoords.support import gauss_2d
from scipy.spatial import distance

#todo r_norm?
def get_coords(model_cell, data_cells, data_elem, r_norm=True):
    dpts = np.sum([np.product(cell.data.shape) for cell in data_cells])
    x = np.empty(dpts, dtype=float)
    y = np.empty(dpts, dtype=float)
    z = np.empty(dpts, dtype=float)

    curr_index = 0
    for cell in data_cells:
        curr_dpts = np.product(cell.data.shape)

        lc = cell.coords.lc / cell.length
        rc = cell.coords.rc * (model_cell.coords.r/cell.coords.r) if r_norm else cell.coords.rc
        _x, _y = model_cell.coords.rev_transform(lc, rc, cell.coords.phi, l_norm=True)
        # _x, _y = rev_transform(cell.coords.lc / cell.length, cell.coords.rc, cell.coords.phi, l_norm=True)

        x[curr_index:curr_dpts + curr_index] = _x.flatten()
        y[curr_index:curr_dpts + curr_index] = _y.flatten()
        z[curr_index:curr_dpts + curr_index] = cell.data.data_dict[data_elem].flatten()

        curr_index += curr_dpts

    return x, y, z


def gauss_kernel(model_cell, x, y, z, sigma=1):
    output = np.empty(model_cell.data.shape)
    coords = np.array([x, y])
    for index in np.ndindex(model_cell.data.shape):
        xi, yi = index
        xp, yp = model_cell.coords.x_coords[xi, yi], model_cell.coords.y_coords[xi, yi]

        dist = distance.cdist(np.array([[xp, yp]]), coords.T).squeeze()
        bools = dist < 5*sigma

        weights = gauss_2d(x[bools], y[bools], xp, yp, sigma=sigma)
        avg = np.average(z[bools], weights=weights)

        output[xi, yi] = avg

    return output