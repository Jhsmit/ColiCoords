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
        print(input_array.dtype)
        assert input_array.dtype in ['int', 'uint', 'uint16', 'uint32']
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
    data_dict = {}
    shape = None

    binary_img = None
    brightfield_img = None
    storm_table = None
    storm_img = None

    def __init__(self, *args, **kwargs):
        pass

    def add_datasets(self, binary_img=None, bf_img=None, fl_data=None, storm_table=None, *args, **kwargs):
        img_data = [binary_img, bf_img] + [v for v in fl_data.values()]
        shapes = [img.shape[:2] for img in img_data if img is not None]
        assert (shapes[1:] == shapes[:-1])
        self.shape = shapes[0] if len(shapes) > 0 else None

        self.binary_img = BinaryImage(binary_img)
        self.brightfield_img = BrightFieldImage(bf_img)

        self.flu_dict = {}
        for k, v in fl_data.items():
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
        if dclass == 'binary':
            assert self.binary_img is None
            self._check_shape(data.shape)
            self.binary_img = BinaryImage(data, name=name, metadata=metadata)
            self.data_dict['binary'] = self.binary_img
        elif dclass == 'brightfield':
            assert self.brightfield_img is None
            self._check_shape(data.shape)
            self.brightfield_img = BrightFieldImage(data, name=name, metadata=metadata)
            self.data_dict['brightfield'] = self.brightfield_img
        elif dclass == 'fluorescence':
            assert name
            assert name not in self.flu_dict
            self._check_shape(data.shape)
            f = FluorescenceImage(data, name=name, metadata=metadata)
            self.flu_dict[name] = f
            setattr(self, 'flu_' + name, f)
        elif dclass == 'storm':
            assert 'storm_table' not in self.data_dict
            self.storm_table = STORMTable(data, name=name, metadata=metadata)
            self.data_dict['storm_table'] = self.storm_table

            assert 'storm_img' not in self.data_dict
            img = self._get_storm_img(data)
            self.storm_img = STORMImage(img, name=name, metadata=metadata)
            self.data_dict['storm_img'] = self.storm_img
        else:
            raise ValueError('Invalid data class')

        self.data_dict.update(self.flu_dict)

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

    def _check_shape(self, shape):
        if self.shape:
            assert shape == self.shape
        else:
            self.shape = shape


    @property
    def dclasses(self):
        return None

