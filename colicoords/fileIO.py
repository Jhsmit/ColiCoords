import h5py
import numpy as np
import os
from colicoords.cell import Cell, CellList
from colicoords.config import cfg
from colicoords.data_models import Data
import re

TYPES = {
    'id': int,
    'frame': int,
    'x': float,
    'y': float,
    'sigma': float,
    'intensity': float,
    'offset': float,
    'bkgstd': float,
    'chi2': float,
    'uncertainty_xy': float
}


#todo add colicoords' version to the files
def save(file_path, cell_obj):
    """
    Save ``ColiCoords`` Cell objects to disk as hdf5-files.

    Parameters
    ----------
    file_path : :obj:`str`
        Target file path.
    cell_obj : :class:`~colicoords.cell.Cell` or :class:`~colicoords.cell.CellList
        ``Cell`` or ``CellList`` object to save to disk.
    """

    if isinstance(cell_obj, Cell):
            with h5py.File(file_path, 'w') as f:
                name = '_cell' if cell_obj.name is None else cell_obj.name
                cell_grp = f.create_group(name)
                _write_cell(cell_grp, cell_obj)

    elif isinstance(cell_obj, CellList):
        with h5py.File(file_path, 'w') as f:
            names = np.array([c.name for c in cell_obj])
            if np.all(names == None):
                names = ['cell_' + str(i).zfill(int(np.ceil(np.log10(len(cell_obj))))) for i in range(len(cell_obj))]
                print('Cell names not defined, assigned default values')
            elif np.any(names == None):
                raise ValueError('Invalidly named cell, either all cells must have a valid name, or none to auto-assign')
            for name, c in zip(names, cell_obj):
                cell_grp = f.create_group(name)
                _write_cell(cell_grp, c)
    else:
        raise ValueError('Invalid type, expected CellList or CellObject, got {}'.format(type(cell_obj)))


def _write_cell(cell_grp, cell_obj):
    """Write a ``Cell`` object to `cell_grp`"""
    attr_grp = cell_grp.create_group('attributes')

    attr_grp.attrs.create('r', cell_obj.coords.r)
    attr_grp.attrs.create('xl', cell_obj.coords.xl)
    attr_grp.attrs.create('xr', cell_obj.coords.xr)
    attr_grp.attrs.create('coeff', cell_obj.coords.coeff)
    attr_grp.attrs.create('name', np.string_(cell_obj.name))

    data_grp = cell_grp.create_group('data')
    for k, v in cell_obj.data.data_dict.items():
        grp = data_grp.create_group(k)
        grp.create_dataset(k, data=v)
        grp.attrs.create('dclass', np.string_(v.dclass))


def _load_cell(cell_grp):
    """Load the cell object from `cell_grp`"""
    data_obj = Data()
    data_grp = cell_grp['data']

    for key in list(data_grp.keys()):
        grp = data_grp[key]
        data_arr = grp[key]
        dclass = grp.attrs.get('dclass').decode('UTF-8')
        data_obj.add_data(data_arr, dclass=dclass, name=key)

    c = Cell(data_obj, init_coords=False)

    attr_grp = cell_grp['attributes']
    attr_dict = dict(attr_grp.attrs.items())

    for a in ['r', 'xl', 'xr', 'coeff']:
        setattr(c.coords, a, attr_dict.get(a))
    c.coords.shape = c.data.shape

    name = attr_dict.get('name').decode('UTF-8')
    c.name = name if name is not 'None' else None

    return c


def load(file_path):
    """
    Load ``Cell`` or ``CellList`` from disk.

    Parameters
    ----------
    file_path : :obj:`str`
        Source file path.

    Returns
    -------
    cell : :class:`~colicoords.cell.Cell` or :class:`~colicoords.cell.CellList`
        Loaded ``Cell`` or ``CellList``
    """

    with h5py.File(file_path, 'r') as f:
        cell_list = [_load_cell(f[key]) for key in f.keys()]

        if len(cell_list) == 1:
            return cell_list[0]
        else:
            return CellList(cell_list)


def load_thunderstorm(file_path, pixelsize=None):
    """
    Load a .csv file from THUNDERSTORM output.

    Parameters
    ----------
    file_path : :obj:`str`
        Target file path to THUNDERSTORM file.
    pixelsize : :obj:`float`, optional
        pixelsize in the THUNDERSTORM file to convert units to pixels. If not specified the default value in config is
        used.

    """

    pixelsize = cfg.IMG_PIXELSIZE if not pixelsize else pixelsize
    ext = os.path.splitext(file_path)[1]
    if ext == '.csv':
        delimiter = ','
    elif ext == '.xls':
        delimiter = '\t'
    else:
        raise ValueError('Invalid data file')

    with open(file_path, 'r') as f:
        line = f.readline()

    names = [re.sub("[\[].*?[\]]", '', s).replace('"', '').strip() for s in line.split(delimiter)]

    dtype = {'names': tuple(names),
             'formats': tuple(TYPES[name] for name in names)}

    storm_table = np.genfromtxt(file_path, skip_header=1, dtype=dtype, delimiter=delimiter)
    storm_table['x'] /= pixelsize
    storm_table['y'] /= pixelsize
    storm_table['uncertainty_xy'] /= pixelsize

    return storm_table


def _load_deprecated(file_path):
    ext = os.path.splitext(file_path)[1]
    if ext == '.cc':
        with h5py.File(file_path, 'r') as f:

            data_obj = Data()
            data_grp = f['data']
            for key in list(data_grp.keys()):
                grp = data_grp[key]
                data_arr = grp[key]
                dclass = grp.attrs.get('dclass').decode('UTF-8')
                data_obj.add_data(data_arr, dclass=dclass, name=key)

            c = Cell(data_obj)

            attr_grp = f['attributes']
            attr_dict = dict(attr_grp.attrs.items())
            for a in ['r', 'xl', 'xr', 'coeff']:
                setattr(c.coords, a, attr_dict.get(a))
            c.name = attr_dict.get('label')

        return c
    else:
        raise ValueError('Invalid file type')