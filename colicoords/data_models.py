import mahotas as mh
import numpy as np
import math
from scipy.ndimage.interpolation import rotate as scipy_rotate
from colicoords.config import cfg


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
        obj.dclass = 'binary'
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
        obj.dclass = 'brightfield'
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
        obj.dclass = 'fluorescence'
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
        obj.dclass = 'storm'
        return obj

    @property
    def image(self):
        raise NotImplementedError()
        #
        # assert 'storm_img' not in self.data_dict
        # img = self._get_storm_img(data)
        # name = 'storm_img' if name is 'storm' else name + '_img'
        # self.storm_img = STORMImage(img, name=name, metadata=metadata)
        # self.data_dict['storm_img'] = self.storm_img
        # self.name_dict[name] = self.storm_img


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
        obj.dclass = 'storm_img'
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
        self.storm_dict = {}

        self.binary_img = None
        self.brightfield_img = None
        self.storm_table = None
        self.storm_img = None

        self.idx = 0
        self.shape = None

    #todo depracate the hell out of this one
    def add_datasets(self, binary_img=None, bf_img=None, flu_data=None, storm_table=None, *args, **kwargs):
        raise DeprecationWarning('NOOOOOOOOOOO')
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
        if dclass == 'binary':
            assert self.binary_img is None
            self._check_shape(data.shape, data.ndim)
            self.binary_img = BinaryImage(data, name=name, metadata=metadata)
            self.data_dict[name] = self.binary_img
        elif dclass == 'brightfield':
            assert self.brightfield_img is None
            self._check_shape(data.shape, data.ndim)
            self.brightfield_img = BrightFieldImage(data, name=name, metadata=metadata)
            self.data_dict[name] = self.brightfield_img
        elif dclass == 'fluorescence':
            assert name
            assert name not in self.flu_dict
            self._check_shape(data.shape, data.ndim)
            f = FluorescenceImage(data, name=name, metadata=metadata)
            setattr(self, 'flu_' + name, f)
            self.flu_dict[name] = f
        elif dclass == 'storm':
            assert name
            assert name not in self.storm_dict
            for field in ['x', 'y', 'frame']:
                assert field in data.dtype.names

        #todo apparently negatie and outside shape values happen and its fine, add prune function.
#             if data['x'].min() < 0:
#                 print(data['x'].min())
# #                raise ValueError('No negative x coordinates allowed')
#             if data['y'].min() < 0:
#                 raise ValueError('No negative y coordinates allowed')
#             if self.shape and self.ndim == 2:
#                 ymax, xmax = self.shape
#                 if data['x'].max() > xmax:
#                     raise ValueError('Storm x coordinate outside of image shape')
#                 if data['y'].max() > ymax:
#                     raise ValueError('Storm y coordinate outside of image shape')
#             elif self.shape and self.ndim == 3:
#                 zmax, ymax, xmax = self.shape
#                 if data['frame'].max() > zmax:
#                     raise ValueError('STORM frame outside of image shape')
#                 if data['x'].max() > xmax:
#                     raise ValueError('Storm x coordinate outside of image shape')
#                 if data['y'].max() > ymax:
#                     raise ValueError('Storm y coordinate outside of image shape')

            s = STORMTable(data, name=name, metadata=metadata)
            self.storm_dict[name] = s
            setattr(self, 'storm_' + name, s)

        else:
            raise ValueError('Invalid data class {}'.format(dclass))

        self.data_dict.update(self.flu_dict)
        self.data_dict.update(self.storm_dict)

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
            if v.dclass == 'storm':
                rotated = _rotate_storm(v, -theta, shape=self.shape)
            elif v.dclass == 'storm_img':
                continue
            else:
                #scipy rotate rotates ccw
                rotated = scipy_rotate(v, -theta)

            data.add_data(rotated, v.dclass, name=v.name, metadata=v.metadata)
        return data

    def transform(self, x, y, src='cart', tgt='mpl'):
        if src == 'cart':
            xt1 = x
            yt1 = y
        elif src == 'mpl':
            xt1 = x
            yt1 = self.shape[0] - y - 0.5
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
            yt2 = self.shape[0] - yt1 - 0.5
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

        #data = Data()
        if self.idx >= len(self):
            self.idx = 0
            raise StopIteration
        else:
            data = self[self.idx]
            self.idx += 1
            return data

    def __getitem__(self, key):
        data = Data()
        print('getitem', key, type(key))
        for v in self.data_dict.values():
            if v.dclass == 'storm':
                # Slicing the STORM data in z-direction
                if type(key) == slice or type(key) == int or len(key) == 3:

                    if type(key) == slice:
                        start, stop, step = key.indices(len(v))
                        selected = np.arange(start, stop, step) + 1
                    elif type(key) == int:
                        selected = np.array([key + 1])
                    else:
                        start, stop, step = key[0].indices(len(v))
                        selected = np.arange(start, stop, step) + 1

                    bools = np.in1d(v['frame'], selected)

                    table_z = v[bools].copy()

                    w = np.where(np.diff(table_z['frame']) != 0)[0]
                    w = np.insert(w, [0, w.size], [-1, len(table_z['frame']) - 1])
                    reps = np.diff(w)
                    new_frames = np.repeat(np.arange(len(reps)) + 1, reps)

                    table_z['frame'] = new_frames

                else:
                    table_z = v

                #XY slicing
                if type(key) == slice or type(key) == int:
                    table_out = table_z
                elif len(key) == 2 or len(key) == 3:
                    print('2d slicing', key, len(key))
                    if len(key) == 2:
                        print('len', len(v))


                        ymin, ymax, ystep = key[0].start, key[0].stop, key[0].step
                        xmin, xmax, xstep = key[1].start, key[1].stop, key[1].step

                        # this doesnt work when len < values
                        # ymin, ymax, ystep = key[0].indices(len(v))
                        # xmin, xmax, xstep = key[1].indices(len(v))
                        print('after', xmin, xmax, ymin, ymax)
                    elif len(key) == 3:
                        ymin, ymax, ystep = key[1].start, key[1].stop, key[1].step
                        xmin, xmax, xstep = key[2].start, key[2].stop, key[2].step

                    print('later', xmin, xmax, ymin, ymax)
                    #Create boolean array to mask entries withing the chosen range
                    b_xy = (table_z['x'] > xmin) * (table_z['x'] < xmax) * (table_z['y'] > ymin) * (table_z['y'] < ymax)
                    print(table_z['x'])
                    print(b_xy)
                    print(np.unique(b_xy))
                # Choose selected data and copy, rezero x and y
              #  b_overall = b_z * b_xy
                    table_out = table_z[b_xy].copy()
                    table_out['x'] -= xmin
                    table_out['y'] -= ymin

                    # print('minmin xy')
                    # print(table_out['x'].min())
                    # print(table_out['y'].min())

                else:
                    print('does this ever occur?')
                    table_out = table_z

                data.add_data(table_out, v.dclass, name=v.name, metadata=v.metadata)

            elif v.dclass == 'storm_img':
                continue
            else:
                data.add_data(v[key], v.dclass, name=v.name, metadata=v.metadata)
        return data

    next = __next__


def _rotate_storm(storm_data, theta, shape=None):
    th_deg = theta
    theta *= np.pi / 180  # to radians
    x = storm_data['x'].copy()
    y = storm_data['y'].copy()

    if shape:
        xmax = shape[0]
        ymax = shape[1]
        offset = 0.5 * shape[0] * ((shape[0]/shape[1]) * np.sin(-theta) + np.cos(-theta) - 1)
        print('OFFSET', offset)

        os2 = - np.cos(np.pi/2 - theta) * shape[1]
        os1 = np.sin(theta) * shape[0] + shape[0] / 2
        print(os2)
        print(os1)
        print(theta, th_deg)
        out_shape = scipy_rotate(np.ones(shape), -th_deg).shape
        os2t = (out_shape[0] - shape[0]) / 2
        os1t = (out_shape[1] - shape[1]) / 2
        print(os2t, os1t)


       # os1 =
    else:
        xmax = int(storm_data['x'].max()) + 2
        ymax = int(storm_data['y'].max()) + 2
        offset = 0

    x -= xmax / 2
    y -= ymax / 2

    xr = x * np.cos(theta) + y * np.sin(theta)
    yr = y * np.cos(theta) - x * np.sin(theta)

    xr += xmax / 2
    xr += os1t
    yr += ymax / 2
    yr -= os2t

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