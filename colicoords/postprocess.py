import numpy as np
from colicoords.support import gauss_2d
from scipy.spatial import distance


def align_cells(model_cell, data_cells, data_name, r_norm=True, sigma=1):
    data_elem = data_cells[0].data.data_dict[data_name]
    if data_elem.dclass == 'fluorescence' or data_elem.dclass == 'brightfield':

        x, y, z = align_images(model_cell, data_cells, data_name, r_norm=r_norm)
        output = gauss_kernel(model_cell, x, y, z, sigma=sigma)
    elif data_elem.dclass == 'storm':
        output = align_storm(model_cell, data_cells, data_name, r_norm=r_norm)
    else:
        raise ValueError('')

    return output


def align_storm(model_cell, data_cells, data_name, r_norm=True):
    dpts = np.sum([len(cell.data.data_dict[data_name]) for cell in data_cells])
    output = np.zeros(dpts, dtype=data_cells[0].data.data_dict[data_name].dtype)

    curr_index = 0
    for cell in data_cells:
        data_elem = cell.data.data_dict[data_name]
        curr_dpts = len(data_elem)

        x, y = data_elem['x'], data_elem['y']
        lc, rc, phi = cell.coords.transform(x, y)
        rc = rc * (model_cell.coords.r/cell.coords.r) if r_norm else rc
        _x, _y = model_cell.coords.rev_transform(lc/cell.length, rc, phi, l_norm=True)

        new_storm = data_elem.copy()
        new_storm['x'] = _x
        new_storm['y'] = _y

        output[curr_index:curr_dpts + curr_index] = new_storm
        curr_index += curr_dpts

    return output


def align_images(model_cell, data_cells, data_name, r_norm=True):
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
        z[curr_index:curr_dpts + curr_index] = cell.data.data_dict[data_name].flatten()

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