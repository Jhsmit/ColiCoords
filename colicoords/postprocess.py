import numpy as np
from colicoords.support import gauss_2d
from scipy.spatial import distance
from tqdm.auto import tqdm
#todo make align cells function which aligns all data elements


def align_data_element(model_cell, data_cells, data_name, r_norm=True, sigma=1):
    """
    Align a data element from a set of cells with respect to the shape of `model_cell`.

    The returned data element has the same shape as the model Cell's data. Returned image data is aligned and averaged
    by a gaussian kernel. Returned STORM data element consists of all aligned individual data element.

    Parameters
    ----------
    model_cell : :class:`~colicoords.cell.Cell`
        Model cell used to align `data_cells` to
    data_cells : :class:`~colicoords.cell.CellList`
        ``CellList`` of data cells to align.
    data_name : :obj:`str`
        Name of the target data element to align.
    r_norm : :obj:`bool`, optional
        Whether or not to normalize the cells with respect to their radius. Default is `True`.
    sigma : :obj:`float`
        Sigma of the gaussian kernel used to calculate output aligned images.

    Returns
    -------
    output : array_like
        Aligned output data element.
    """
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
    """
    Align STORM data element with respect to the shape of `model_cell`.

    The returned STORM data element consists of all aligned individual data element.

    Parameters
    ----------
    model_cell : :class:`~colicoords.cell.Cell`
        Model cell used to align `data_cells` to
    data_cells : :class:`~colicoords.cell.CellList`
        ``CellList`` of data cells to align.
    data_name : :obj:`str`
        Name of the target STORM data element to align.
    r_norm : :obj:`bool`, optional
        Whether or not to normalize the cells with respect to their radius. Default is `True`.

    Returns
    -------
    output : array_like
        Aligned output data element.
    """

    dpts = np.sum([len(cell.data.data_dict[data_name]) for cell in data_cells])
    output = np.zeros(dpts, dtype=data_cells[0].data.data_dict[data_name].dtype)

    curr_index = 0
    for cell in tqdm(data_cells, desc='Align STORM'):
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
    """
    Align image data with respect to the shape of `model_cell`.

    The returned data element has the same shape as the model Cell's data. The returned image data is aligned and
    averaged by a gaussian kernel.

    Parameters
    ----------
    model_cell : :class:`~colicoords.cell.Cell`
        Model cell used to align `data_cells` to.
    data_cells : :class:`~colicoords.cell.CellList`
        ``CellList`` of data cells to align.
    data_name : :obj:`str`
        Name of the target data element to align.
    r_norm : :obj:`bool`, optional
        Whether or not to normalize the cells with respect to their radius. Default is `True`.

    Returns
    -------
    x : :class:`~numpy.ndarray`
        Array with combined x-coordinates of aligned pixels.
    y : :class:`~numpy.ndarray`
        Array with combined y-coordinates of aligned pixels.
    z : :class:`~numpy.ndarray`
        Array with pixel values of aligned pixels.
    """

    dpts = np.sum([np.product(cell.data.shape) for cell in data_cells])
    x = np.empty(dpts, dtype=float)
    y = np.empty(dpts, dtype=float)
    z = np.empty(dpts, dtype=float)

    curr_index = 0
    for cell in tqdm(data_cells, desc='Align images'):
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
    """
    Convolute aligned pixels given coordinates `x`, `y` and values `z` with a gaussian kernel to form the final image.

    Parameters
    ----------
    model_cell : :class:`~colicoords.cell.Cell`
        Model cell defining output shape.
    x : :class:`~numpy.ndarray`
        Array with combined x-coordinates of aligned pixels.
    y : :class:`~numpy.ndarray`
        Array with combined y-coordinates of aligned pixels.
    z : :class:`~numpy.ndarray`
        Array with pixel values of aligned pixels.
    sigma : :obj:`float`
        Sigma of the gaussian kernel.

    Returns
    -------
    output : :class:`~numpy.ndarray`
        Output aligned image.
    """

    output = np.empty(model_cell.data.shape)
    coords = np.array([x, y])
    for index in tqdm(np.ndindex(model_cell.data.shape), desc='Gaussian kernel', total=np.product(model_cell.data.shape)):
        xi, yi = index
        xp, yp = model_cell.coords.x_coords[xi, yi], model_cell.coords.y_coords[xi, yi]

        dist = distance.cdist(np.array([[xp, yp]]), coords.T).squeeze()
        bools = dist < 5*sigma

        weights = gauss_2d(x[bools], y[bools], xp, yp, sigma=sigma)
        avg = np.average(z[bools], weights=weights)

        output[xi, yi] = avg

    return output
