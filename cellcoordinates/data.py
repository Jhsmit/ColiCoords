import mahotas as mh
import numpy as np
import math
from scipy.ndimage.interpolation import rotate as scipy_rotate
from config import cfg


class BinaryImage(np.ndarray):
    def __new__(cls, input_array, name=None, metadata=None):
        if input_array is None:
            return None

        #bool_arr = input_array.astype(bool)
        #assert np.array_equal(input_array, bool_arr)
        #either boolean or labeled binary
        assert input_array.dtype in ['int', 'uint', 'uint16', 'uint32', 'bool']
        #todo test connectedness l, n = mh.label(bool_arr)

        #todo binary is now saved as int

        obj = np.asarray(input_array).view(cls)
        obj.name = name
        obj.metadata = metadata
        obj.dclass = 'Binary'
        return obj


class BrightFieldImage(np.ndarray):
    def __new__(cls, input_array, name=None, metadata=None):
        if input_array is None:
            return None
        obj = np.asarray(input_array).view(cls)
        obj.name = name
        obj.metadata = metadata
        obj.dclass = 'Brightfield'
        return obj


class FluorescenceImage(np.ndarray):
    def __new__(cls, input_array, name=None, metadata=None):
        if input_array is None:
            return None
        obj = np.asarray(input_array).view(cls)
        obj.name = name
        obj.metadata = metadata
        obj.dclass = 'Fluorescence'
        return obj


class STORMTable(np.ndarray):
    """STORM data array
    
    Args:
        input_array: 
    """

    def __new__(cls, input_array, name=None, metadata=None):

        if input_array is None:
            return None
        obj = np.asarray(input_array).view(cls)
        obj.name = name
        obj.metadata = metadata
        obj.dclass = 'STORMTable'
        return obj


class STORMImage(np.ndarray):
    def __new__(cls, input_array, name=None, metadata=None):
        """STORM recontructed image
        
        Args:
            input_array: STORM data array 
        """
        if input_array is None:
            return None

        obj = np.asarray(input_array).view(cls)
        obj.name = name
        obj.metadata = metadata
        obj.dclass = 'STORMImage'
        return obj

#todo this shoud be a dict? (use open microscopy format?) (XML)
class MetaData(dict):
    pass


class Data(object):
    """
    Parent object for all data classes
    """

    flu_dict = {}
    data_dict = {}  #stores by dtype or name
    name_dict = {}  #stores by name
    shape = None

    binary_img = None
    brightfield_img = None
    storm_table = None
    storm_img = None

    idx = 0


    def __init__(self, *args, **kwargs):
        pass

    def add_datasets(self, binary_img=None, bf_img=None, flu_data=None, storm_table=None, *args, **kwargs):
        img_data = [binary_img, bf_img] + [v for v in flu_data.values()]
        shapes = [img.shape[:2] for img in img_data if img is not None]
        assert (shapes[1:] == shapes[:-1])
        self.shape = shapes[0] if len(shapes) > 0 else None

        self.binary_img = BinaryImage(binary_img)
        self.brightfield_img = BrightFieldImage(bf_img)

        self.flu_dict = {}
        for k, v in flu_data.items():
                d = FluorescenceImage(v)
                self.flu_dict[k] = d
                setattr(self, 'flu_' + k, d)

        self.storm_table = STORMTable(storm_table)

        if storm_table is not None:
            if self.shape:
                xmax = self.shape[0] * cfg.IMG_PIXELSIZE
                ymax = self.shape[1] * cfg.IMG_PIXELSIZE
            else:
                xmax = int(storm_table['x'].max()) + 2 * cfg.STORM_PIXELSIZE
                ymax = int(storm_table['y'].max()) + 2 * cfg.STORM_PIXELSIZE

            x_bins = np.arange(0, xmax, cfg.STORM_PIXELSIZE)
            y_bins = np.arange(0, ymax, cfg.STORM_PIXELSIZE)

            h, xedges, yedges = np.histogram2d(storm_table['x'], storm_table['y'], bins=[x_bins, y_bins])

            self.storm_img = STORMImage(h.T)

        self.data_dict = {'binary': self.binary_img,
                          'brightfield': self.brightfield_img,
                          'storm_table': self.storm_table}
        self.data_dict.update(self.flu_dict)

    def add_data(self, data, dclass, name=None, metadata=None):
        dclass = dclass.lower()
        if name is None:
            name = dclass
        assert name not in [d.name for d in self.data_dict.values()]
        if dclass == 'binary':
            assert self.binary_img is None
            self._check_shape(data.shape, data.ndim)
            self.binary_img = BinaryImage(data, name=name, metadata=metadata)
            self.data_dict['binary'] = self.binary_img
            name = 'binary' if not name else name
            self.name_dict[name] = self.binary_img
        elif dclass == 'brightfield':
            assert self.brightfield_img is None
            self._check_shape(data.shape, data.ndim)
            self.brightfield_img = BrightFieldImage(data, name=name, metadata=metadata)
            self.data_dict['brightfield'] = self.brightfield_img
            name = 'brightfield' if not name else name
            self.name_dict[name] = self.brightfield_img
        elif dclass == 'fluorescence':
            assert name
            assert name not in self.flu_dict
            self._check_shape(data.shape, data.ndim)
            f = FluorescenceImage(data, name=name, metadata=metadata)
            self.flu_dict[name] = f
            setattr(self, 'flu_' + name, f)
        elif dclass == 'storm':
            #todo some checks to make sure there is a frame entry in the table when ndim == 3
            assert 'storm_table' not in self.data_dict
            self.storm_table = STORMTable(data, name=name, metadata=metadata)
            self.data_dict['storm_table'] = self.storm_table
            name = 'storm_table' if not name else name
            self.name_dict[name] = self.storm_table


            assert 'storm_img' not in self.data_dict
            img = self._get_storm_img(data)
            name = 'storm_img' if not name else name + '_img'
            self.storm_img = STORMImage(img, name=name, metadata=metadata)
            self.data_dict['storm_img'] = self.storm_img
            self.name_dict[name] = self.storm_img
        else:
            raise ValueError('Invalid data class')

        self.data_dict.update(self.flu_dict)
        self.name_dict.update(self.flu_dict)

    def _get_storm_img(self, storm_table):
        if self.shape:
            xmax = self.shape[0] * cfg.IMG_PIXELSIZE
            ymax = self.shape[1] * cfg.IMG_PIXELSIZE
        else:
            xmax = int(storm_table['x'].max()) + 2 * cfg.STORM_PIXELSIZE
            ymax = int(storm_table['y'].max()) + 2 * cfg.STORM_PIXELSIZE

        x_bins = np.arange(0, xmax, cfg.STORM_PIXELSIZE)
        y_bins = np.arange(0, ymax, cfg.STORM_PIXELSIZE)

        h, xedges, yedges = np.histogram2d(storm_table['x'], storm_table['y'], bins=[x_bins, y_bins])

        return h.T

    def _check_shape(self, shape, ndim):
        if self.shape:
            assert shape == self.shape
            assert ndim == self.ndim
        else:
            self.shape = shape
            self.ndim = ndim

    def from_name(self, name):
        idx

    @property
    def dclasses(self):
        return np.unique([d.dclass for d in self.data_dict.values()])

    @property
    def names(self):
        return [d.name for d in self.data_dict.values()]

    def __len__(self):
        if self.ndim == 3:
            return self.shape[0]
        elif self.ndim == 2:
            return 1
        else:
            raise ValueError

    def __next__(self):
        if self.ndim == 2:
            self.idx = 0
            raise StopIteration
        data = Data()
        for v in self.data_dict.values():
            data.add_data(v[self.idx], v.dclass, name=v.name, metadata=v.metadata)
        self.idx += 1
        if self.idx >= self.length:
            self.idx = 0
            raise StopIteration
        else:
            return data


    next = __next__
