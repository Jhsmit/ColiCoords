import mahotas as mh
import numpy as np
import math
from scipy.ndimage.interpolation import rotate as scipy_rotate
from cellcoordinates.config import cfg


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

    @property
    def orientation(self):
        return _calc_orientation(self)


class BrightFieldImage(np.ndarray):
    def __new__(cls, input_array, name=None, metadata=None):
        if input_array is None:
            return None
        obj = np.asarray(input_array).view(cls)
        obj.name = name
        obj.metadata = metadata
        obj.dclass = 'Brightfield'
        return obj

    @property
    def orientation(self):
        return _calc_orientation(self)


class FluorescenceImage(np.ndarray):
    def __new__(cls, input_array, name=None, metadata=None):
        if input_array is None:
            return None
        obj = np.asarray(input_array).view(cls)
        obj.name = name
        obj.metadata = metadata
        obj.dclass = 'Fluorescence'
        return obj

    @property
    def orientation(self):
        return _calc_orientation(self)


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

    @property
    def orientation(self):
        return _calc_orientation(self)


#todo this shoud be a dict? (use open microscopy format?) (XML)
class MetaData(dict):
    pass


class Data(object):
    """
    Parent object for all data classes
    """

    def __init__(self, *args, **kwargs):
        self.data_dict = {}
        self.flu_dict = {}  #needed or new initialized class doesnt have empty dicts!!!oneone
        self.name_dict = {}

        self.binary_img = None
        self.brightfield_img = None
        self.storm_table = None
        self.storm_img = None

        self.idx = 0
        self.shape = None

    #todo depracate the hell out of this one
    def add_datasets(self, binary_img=None, bf_img=None, flu_data=None, storm_table=None, *args, **kwargs):
        flu_data = {} if flu_data == None else flu_data
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

        #todoo old names
        self.data_dict = {'Binary': self.binary_img,
                          'Brightfield': self.brightfield_img,
                          'STORMTable': self.storm_table}
        self.data_dict.update(self.flu_dict)

    def add_data(self, data, dclass, name=None, metadata=None):
        if name in ['', u'', r'', None]:
            name = dclass
        else:
            name = str(name)

        assert name not in [d.name for d in self.data_dict.values()]
        if dclass == 'Binary':
            assert self.binary_img is None
            self._check_shape(data.shape, data.ndim)
            self.binary_img = BinaryImage(data, name=name, metadata=metadata)
            self.data_dict[name] = self.binary_img
        elif dclass == 'Brightfield':
            assert self.brightfield_img is None
            self._check_shape(data.shape, data.ndim)
            self.brightfield_img = BrightFieldImage(data, name=name, metadata=metadata)
            self.data_dict[name] = self.brightfield_img
        elif dclass == 'Fluorescence':
            assert name
            assert name not in self.flu_dict
            self._check_shape(data.shape, data.ndim)
            f = FluorescenceImage(data, name=name, metadata=metadata)
            setattr(self, 'flu_' + name, f)
            self.flu_dict[name] = f
        elif dclass == 'STORMTable':
            #todo some checks to make sure there is a frame entry in the table when ndim == 3 and vice versa
            assert 'STORMTable' not in self.data_dict
            self.storm_table = STORMTable(data, name=name, metadata=metadata)
            self.data_dict['STORMTable'] = self.storm_table
            self.name_dict[name] = self.storm_table

            assert 'STORMImage' not in self.data_dict
            img = self._get_storm_img(data)
            name = 'STORMImage' if name is 'STORMTable' else name + '_img'
            self.storm_img = STORMImage(img, name=name, metadata=metadata)
            self.data_dict['STORMImage'] = self.storm_img
            self.name_dict[name] = self.storm_img
        else:
            raise ValueError('Invalid data class')

        self.data_dict.update(self.flu_dict)
        self.name_dict.update(self.flu_dict)

    def copy(self):
        data = Data()
        for v in self.data_dict.values():
            data.add_data(np.copy(v), v.dclass, name=v.name, metadata=v.metadata)
        return data



    @property
    def dclasses(self):
        return np.unique([d.dclass for d in self.data_dict.values()])

    @property
    def names(self):
        return [d.name for d in self.data_dict.values()]

    def rotate(self, theta):
        data = Data()
        for v in self.data_dict.values():
            if v.dclass == 'STORMTable':
                rotated = _rotate_storm(v, -theta)
            elif v.dclass == 'STORMImage':
                continue
            else:
                rotated = scipy_rotate(v, -theta)

            data.add_data(rotated, v.dclass, name=v.name, metadata=v.metadata)
        return data

    def transform(self, x, y, src='cart', tgt='mpl'):
        if src == 'cart':
            xt1 = x
            yt1 = y
        elif src == 'mpl':
            xt1 = x
            yt1 = self.shape[0] - y
        elif src == 'matrix':
            yt1 = self.shape[0] - x - 0.5
            xt1 = y + 0.5
        else:
            raise ValueError("Invalid source coordinates")

        if tgt == 'cart':
            xt2 = xt1
            yt2 = yt1
        elif tgt == 'mpl':
            xt2 = xt1
            yt2 = self.shape[0] - yt1
        elif tgt == 'matrix':
            xt2 = self.shape[0] - yt1 - 0.5
            yt2 = xt1 - 0.5
        else:
            raise ValueError("Invalid target coordinates")
        return xt2, yt2

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

    def __len__(self):
        if not hasattr(self, 'ndim'):
            return 0
        elif self.ndim == 3:
            return self.shape[0]
        elif self.ndim == 2:
            return 1
        else:
            raise ValueError

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if not hasattr(self, 'ndim'):
            raise StopIteration
        if self.ndim == 2:
            if self.idx == 0:
                self.idx += 1
                return self
            else:
                self.idx = 0
                raise StopIteration

        data = Data()
        for v in self.data_dict.values():
            data.add_data(v[self.idx], v.dclass, name=v.name, metadata=v.metadata)
        self.idx += 1
        if self.idx >= len(self):
            self.idx = 0
            raise StopIteration
        else:
            return data

    def __getitem__(self, key):
        data = Data()
        for v in self.data_dict.values():
            if v.dclass == 'STORMTable':
                b_z = np.ones(len(v)).astype(bool)
                if len(key) == 3:
                    #3d slicing, slices the frames? #todo 3d slicing by frame!
                    raise NotImplementedError()

                elif len(key) == 2:
                    ymin, ymax, ystep = key[0].indices(len(v))
                    xmin, xmax, ystep = key[1].indices(len(v))

                    ymin *= cfg.IMG_PIXELSIZE
                    ymax *= cfg.IMG_PIXELSIZE
                    xmin *= cfg.IMG_PIXELSIZE
                    xmax *= cfg.IMG_PIXELSIZE

                    #Create boolean array to mask entries withing the chosen range
                    b_xy = (v['x'] > xmin) * (v['x'] < xmax) * (v['y'] > ymin) * (v['y'] < ymax)

                # Choose selected data and copy, rezero x and y
                b_overall = b_z * b_xy
                table_out = v[b_overall].copy()
                table_out['x'] -= xmin
                table_out['y'] -= ymin

                data.add_data(table_out, v.dclass, name=v.name, metadata=v.metadata)

            elif v.dclass == 'STORMImage':
                continue
            else:
                data.add_data(v[key], v.dclass, name=v.name, metadata=v.metadata)
        return data

    next = __next__


def _rotate_storm(storm_data, theta, shape=None):
    theta *= np.pi / 180  # to radians
    x = storm_data['x']
    y = storm_data['y']

    if shape:
        xmax = shape[0] * cfg.IMG_PIXELSIZE
        ymax = shape[1] * cfg.IMG_PIXELSIZE
    else:
        xmax = int(storm_data['x'].max()) + 2 * cfg.STORM_PIXELSIZE
        ymax = int(storm_data['y'].max()) + 2 * cfg.STORM_PIXELSIZE

    x -= xmax / 2
    y -= ymax / 2

    xr = x * np.cos(theta) + y * np.sin(theta)
    yr = y * np.cos(theta) - x * np.sin(theta)

    xr += xmax / 2
    yr += ymax / 2

    storm_out = np.copy(storm_data)
    storm_out['x'] = xr
    storm_out['y'] = yr

    return storm_out


def _calc_orientation(img):
    com = mh.center_of_mass(img)

    mu00 = mh.moments(img, 0, 0, com)
    mu11 = mh.moments(img, 1, 1, com)
    mu20 = mh.moments(img, 2, 0, com)
    mu02 = mh.moments(img, 0, 2, com)

    mup_20 = mu20 / mu00
    mup_02 = mu02 / mu00
    mup_11 = mu11 / mu00

    theta_rad = 0.5 * math.atan(2 * mup_11 / (mup_20 - mup_02))  # todo math -> numpy
    theta = theta_rad * (180 / math.pi)
    if (mup_20 - mup_02) > 0:
        theta += 90

    return theta