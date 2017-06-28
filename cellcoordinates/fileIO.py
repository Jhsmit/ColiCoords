import h5py
import numpy as np
from cell import Cell


def save(file_path, cell_obj):
    with h5py.File(file_path, 'w') as f:
        attr_grp = f.create_group('attributes')

        attr_grp.attrs.create('r', cell_obj.coords.r)
        attr_grp.attrs.create('xl', cell_obj.coords.xl)
        attr_grp.attrs.create('xr', cell_obj.coords.xr)
        attr_grp.attrs.create('coeff', cell_obj.coords.coeff)

        #todo python 3 compatiblity: https://github.com/h5py/h5py/issues/441
        attr_grp.attrs.create('label', cell_obj.label.encode())

        data_grp = f.create_group('data')
        for k, v in cell_obj.data.data_dict.items():
            if v is not None:
                data_grp.create_dataset(k, data=v)


def load(file_path):
    with h5py.File(file_path, 'r') as f:

        data_grp = f['data']
        data_dict = {item: np.array(data_grp.get(item)) for item in data_grp}

        c = Cell(data_dict=data_dict)

        attr_grp = f['attributes']
        attr_dict = dict(attr_grp.attrs.items())
        for a in ['r', 'xl', 'xr', 'coeff']:
            setattr(c.coords, a, attr_dict.get(a))
        c.label = attr_dict.get('label')

    return c