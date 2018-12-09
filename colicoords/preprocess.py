import mahotas as mh
import numpy as np
from numpy.lib.polynomial import RankWarning
import warnings
from colicoords.cell import Cell, CellList


def filter_binaries(bin_arr, remove_bordering=True, min_size=None, max_size=None, max_minor=None, max_major=None):
    """
    Filters and labels a stack of binary images.

    Parameters
    ----------
    bin_arr : :class:`~numpy.ndarray`
        Input binary array.
    remove_bordering : :obj:`bool`
        Remove regions at the image border.
    min_size : :obj:`int`
        Minimum size of binary regions. ``None`` to ignore.
    max_size : :obj:`int`
        Maximum size of binary regions. ``None`` to ignore.
    max_minor : :obj:`int`
        Maximum length of the semiminor ellipse axes of the binary region. ``None`` to ignore.
    max_major : :obj:`int`
        Maximum length of the simimajor ellipse axes of the binary region. ``None`` to ignore.

    Returns
    -------
    out : :class:`~numpy.ndarray`
        Output filtered and labeled binary image.
    """

    out = np.empty_like(bin_arr)
    for i, img in enumerate(bin_arr):
        labeled, n = mh.labeled.label(img)
        labeled, n = mh.labeled.filter_labeled(labeled, remove_bordering=remove_bordering, min_size=min_size, max_size=max_size)
        out[i] = labeled

    for j, img in enumerate(out):
        for i in np.unique(img)[1:]:
            selected_binary = (img == i).astype('int')
            min1, max1, min2, max2 = mh.bbox(selected_binary)
            selection = img[min1:max1, min2:max2]
            major, minor = mh.features.ellipse_axes(selection)

            if max_minor and minor > max_minor:
                img[img == i] = 0
            if max_major and major > max_major:
                img[img == i] = 0

    return out


#todo split into filter binary and data to cells
def data_to_cells(input_data, initial_crop=5, final_crop=7, rotate='binary', remove_bordering=True,
                  remove_multiple_cells=True, remove_poor_fit=True, init_coords=True, verbose=True):
    """
    Create ``Cell`` objects from input ``Data`` object.

    Cell are identified in each frame by the binary image and all the data is cropped across all data elements to
    bundle the data for one cell together in a :class:`~colicoords.cell.Cell` object.

    Parameters
    ----------
    input_data :class:`~colicoords.data_models.Data
        Data object with input data. Must be a 3D data object with a labelled binary image.
    initial_crop : :obj:`int`
        Number of pixels around the binary image of the cell to crop the image (on all sides).
    final_crop : :obj:`int`
        Number of pixels around the rotated binary image of the cell to crop the image (on all sides)
    rotate : :obj:`str`
        Name of the data element to use to calculate the cell's orientation. The cell will be rotated to orient it
        horizontally. If `None` the cell is not rotated.
    remove_bordering : :obj:`bool`
        If `True` cells at the border will not be added.
    remove_multiple_cells : :obj:`bool`
        If `True` when a selection is made around a cell object but this selection contains another cell, it is skipped.
    remove_poor_fit : :obj:`bool`
        If `True` when initializing coordinate system for the cell raises a ``RankWarning`` due to poor polyfit, the
        cell is skipped.
    init_coords : :obj:`bool`
        If `False` the coordinate system of the ``Cell`` objects will not be initialized.
    verbose : :obj:`bool`
        If `True` the method is ran in verbose mode, set to `False` to disable.

    Returns
    -------
    cell_list : :class:`CellList`:
        List of  :class:`~colicoords.cell.Cell` objects
    """

    assert 'binary' in input_data.dclasses
    assert input_data.ndim == 3

    vprint = print if verbose else lambda *a, **k: None
    cell_list = []
    i_fill = int(np.ceil(np.log10(len(input_data))))
    for i, data in enumerate(input_data):
        binary = data.binary_img
        if binary.mean() == 0.:
            vprint('Image {} {}: No cells'.format(binary.name, i))
            continue

        # Iterate over all cells in the image
        l_fill = int(np.ceil(np.log10(len(np.unique(binary)))))
        for l in np.unique(binary)[1:]:
            selected_binary = (binary == l).astype('int')
            min1, max1, min2, max2 = mh.bbox(selected_binary)
            min1p, max1p, min2p, max2p = min1 - initial_crop, max1 + initial_crop, min2 - initial_crop, max2 + initial_crop

            if remove_bordering:
                try:
                    assert min1p > 0 and min2p > 0 and max1p < data.shape[0] and max2p < data.shape[1]
                except AssertionError:
                    vprint('Cell {} on image {} {}: on the edge of the image'.format(l, binary.name, i))
                    continue

            if remove_multiple_cells:
                try:
                    assert len(np.unique(binary[min1p:max1p, min2p:max2p])) == 2
                except AssertionError:
                    vprint('Cell {} on image {} {}: multiple cells per selection'.format(l, data.binary_img.name, i))
                    continue

            # if bordering are not remove some indices might be outside of the image dimensions
            min1f = np.max((min1p, 0))
            max1f = np.min((max1p, data.shape[0]))

            min2f = np.max((min2p, 0))
            max2f = np.min((max2p, data.shape[1]))

            output_data = data[min1f:max1f, min2f:max2f].copy()
            output_data.data_dict['binary'] = (output_data.binary_img == l).astype(int)

            # Calculate rotation angle and rotate selections
            if rotate:
                theta = output_data.data_dict[rotate].orientation
                if theta % 45 == 0:
                    theta += 90
                rotated_data = output_data.rotate(theta)

                try:
                    assert np.abs(rotated_data.binary_img.orientation) < 10
                except AssertionError:
                    vprint('Cell {} on image {} {}: invalid orientation'.format(l, data.binary_img.name, i))
                    pass
            else:
                rotated_data = output_data

            if final_crop:
                min1, max1, min2, max2 = mh.bbox(rotated_data.binary_img)
                min1p, max1p, min2p, max2p = min1 - final_crop, max1 + final_crop, min2 - final_crop, max2 + final_crop

                min1f = np.max((min1p, 0))
                max1f = np.min((max1p, rotated_data.shape[0]))

                min2f = np.max((min2p, 0))
                max2f = np.min((max2p, rotated_data.shape[1]))
                #todo acutal padding instead of crop?

                final_data = rotated_data[min1f:max1f, min2f:max2f].copy()
            else:
                final_data = rotated_data
            #Make cell object and add all the data
            #todo change cell initation and data adding interface

            if remove_poor_fit:
                #Might want to move this context manager outside of for loop
                with warnings.catch_warnings(record=True) as w:
                    c = Cell(final_data, init_coords=init_coords)
                if w and w[0].category == RankWarning:
                    continue
            else:
                c = Cell(final_data, init_coords=init_coords)

            c.name = 'img{}c{}'.format(str(i).zfill(i_fill), str(l).zfill(l_fill))
            cell_list.append(c)

    return CellList(cell_list)


def data_to_cell_lists(input_data, initial_crop=5, final_crop=7, rotate='binary'):
    """
    Create a list of :class:`CellList` from the input :class:`Data` object.

    Typically used for time-lapse data.

    Parameters
    ----------
    input_data (:class:`Data`): Data object with input data. Must be a 3D data object with a labelled binary image,
        and every frame must have the same amount of labelled objects.
    initial_crop : :obj:`int`
        Number of pixels around the binary image of the cell to crop the image (on all sides).
    final_crop : :obj:`int`
        Number of pixels around the rotated binary image of the cell to crop the image (on all sides)
    rotate : :obj:`str`
        Name of the data element to use to calculate the cell's orientation. The cell will be rotated to orient it
        horizontally. If `None` the cell is not rotated.

    Returns
    -------
    cell_list_list : :obj:`list`
        List of :class:`CellList` objects
    """

    assert 'binary' in input_data.dclasses
    assert input_data.ndim == 3

    cell_list_list = []
    labels = np.unique(input_data.binary_img[0])[1:]
    f_fill = int(np.ceil(np.log10(len(input_data))))
    c_fill = int(np.ceil(np.log10(len(labels))))

    for label in labels:
        cell_list = CellList([])
        for i, bin_img in enumerate(input_data.binary_img):
            assert np.all(np.unique(bin_img)[1:] == labels)
            selected_binary = (bin_img == label).astype('int')

            min1, max1, min2, max2 = mh.bbox(selected_binary)
            min1p, max1p, min2p, max2p = min1 - initial_crop, max1 + initial_crop, min2 - initial_crop, max2 + initial_crop
            min1p = np.max([0, min1p])
            min2p = np.max([0, min2p])
            max1p = np.min([selected_binary.shape[0], max1p])
            max2p = np.min([selected_binary.shape[1], max2p])

            output_data = input_data[i, min1p:max1p, min2p:max2p].copy()
            output_data.binary_img = (output_data.binary_img == label).astype(int)

            theta = output_data.data_dict[rotate].orientation if rotate else 0
            rotated_data = output_data.rotate(theta)

            if final_crop:
                min1, max1, min2, max2 = mh.bbox(rotated_data.binary_img)
                min1p, max1p, min2p, max2p = min1 - final_crop, max1 + final_crop, min2 - final_crop, max2 + final_crop
                min1f = np.max((min1p, 0))
                max1f = np.min((max1p, rotated_data.shape[0]))

                min2f = np.max((min2p, 0))
                max2f = np.min((max2p, rotated_data.shape[1]))
                # todo acutal padding instead of crop?

                final_data = rotated_data[min1f:max1f, min2f:max2f].copy()
            else:
                final_data = rotated_data
                # Make cell object and add all the data
                # todo change cell initation and data adding interface
            name = 'frame{}c{}'.format(str(i).zfill(f_fill), str(label).zfill(c_fill))
            c = Cell(final_data, name=name)

            cell_list.append(c)

        cell_list_list.append(cell_list)

    return cell_list_list