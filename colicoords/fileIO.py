import h5py
import numpy as np
import tifffile
import os
from xml.etree import cElementTree as etree
import warnings
from colicoords.cell import Cell, CellList
from colicoords.config import cfg
from colicoords.data_models import Data

#todo add colicoords' version to the files
def save(file_path, cell_obj, imagej=False):
    ext = os.path.splitext(file_path)[1]

    if isinstance(cell_obj, Cell):
        if ext == '.cc' or '':
            if ext == '':
                file_path += '.cc'

            with h5py.File(file_path, 'w') as f:
                name = '_cell' if cell_obj.name is None else cell_obj.name
                cell_grp = f.create_group(name)
                _write_cell(cell_grp, cell_obj)

        elif ext == '.tif' or '.tiff':
            raise NotImplementedError()
            # with tifffile.TiffWriter(file_path, imagej=imagej) as tif:
            #     for k, v in cell_obj.data.data_dict.items():
            #         if v is not None:
            #             if imagej:
            #                 if v.dtype == 'int32':
            #                     print(k)
            #                     print('something is int32')
            #                     v = v.astype('int16')
            #
            #             tif.save(v.astype('int32'), description=k)

    elif isinstance(cell_obj, CellList):
        if ext == 'cc' or '':
            if ext == '':
                file_path += '.cc'

        with h5py.File(file_path, 'w') as f:
            for c in cell_obj:
                name = 'None' if c.name is None else c.name
                cell_grp = f.create_group(name)
                _write_cell(cell_grp, c)

    else:
        raise ValueError('Invalid type, expected CellList or CellObject, got {}'.format(type(cell_obj)))


def _write_cell(cell_grp, cell_obj):
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
    data_obj = Data()
    data_grp = cell_grp['data']

    for key in list(data_grp.keys()):
        grp = data_grp[key]
        data_arr = grp[key]
        dclass = grp.attrs.get('dclass').decode('UTF-8')
        data_obj.add_data(data_arr, dclass=dclass, name=key)

    c = Cell(data_obj)

    attr_grp = cell_grp['attributes']
    attr_dict = dict(attr_grp.attrs.items())

    for a in ['r', 'xl', 'xr', 'coeff']:
        setattr(c.coords, a, attr_dict.get(a))

    name = attr_dict.get('name').decode('UTF-8')
    c.name = name if name is not 'None' else None

    return c


def load(file_path):
    ext = os.path.splitext(file_path)[1]

    if ext == '.cc':
        with h5py.File(file_path, 'r') as f:
            cell_list = [_load_cell(f[key]) for key in f.keys()]

            if len(cell_list) == 1:
                return cell_list[0]
            else:
                return CellList(cell_list)


            # data_obj = Data()
            # data_grp = f['data']
            # for key in list(data_grp.keys()):
            #     grp = data_grp[key]
            #     data_arr = grp[key]
            #     dclass = grp.attrs.get('dclass').decode('UTF-8')
            #     data_obj.add_data(data_arr, dclass=dclass, name=key)
            #
            # c = Cell(data_obj)
            #
            # attr_grp = f['attributes']
            # attr_dict = dict(attr_grp.attrs.items())
            # for a in ['r', 'xl', 'xr', 'coeff']:
            #     setattr(c.coords, a, attr_dict.get(a))
            # c.name = attr_dict.get('label')

    else:
        raise ValueError('Invalid file type')

    # elif ext == '.tif' or '.tiff':
    #     with tifffile.TiffFile(file_path, is_ome=True) as tif:
    #         omexml = tif.pages[0].tags['image_description'].value
    #         try:
    #             root = etree.fromstring(omexml)
    #         except etree.ParseError as e:
    #             # TODO: test this
    #             warnings.warn("ome-xml: %s" % e)
    #             omexml = omexml.decode('utf-8', 'ignore').encode('utf-8')
    #             root = etree.fromstring(omexml)


def load_thunderstorm(file_path, pixelsize=None):
    """
    Load a .csv file from THUNDERSTORM output
    :param file_path: Target file to open
    :return:
    """

    assert(os.path.splitext(file_path)[1] == '.csv')

    dtype = {
        'names': ("id", "frame", "x", "y", "sigma", "intensity", "offset", "bkgstd", "chi2", "uncertainty_xy"),
        'formats': (int, int, float, float, float, float, float, float, float, float)
    }

    pixelsize = cfg.IMG_PIXELSIZE if not pixelsize else pixelsize

    storm_table = np.genfromtxt(file_path, skip_header=1, dtype=dtype, delimiter=',')
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