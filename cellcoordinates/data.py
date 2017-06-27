import mahotas as mh
import numpy as np
import math
from scipy.ndimage.interpolation import rotate as scipy_rotate


class DataBaseClass(object):
    pass


class BinaryImage(np.ndarray):
    def __new__(cls, input_array, label=None, metadata=None):
        if input_array is None:
            return None

        bool_arr = input_array.astype(bool)
        assert np.array_equal(input_array, bool_arr)
        #todo test connectedness l, n = mh.label(bool_arr)

        obj = np.asarray(bool_arr).view(cls)
        obj.label = label
        obj.metadata = metadata
        return obj


class BrightFieldImage(np.ndarray):
    def __new__(cls, input_array, label=None, metadata=None):
        if input_array is None:
            return None
        obj = np.asarray(input_array).view(cls)
        obj.label = label
        obj.metadata = metadata
        return obj


class FluorescenceImage(np.ndarray):
    def __new__(cls, input_array, label=None, metadata=None):
        if input_array is None:
            return None
        obj = np.asarray(input_array).view(cls)
        obj.label = label
        obj.metadata = metadata
        return obj


class FluorescenceMovie(np.ndarray):
    def __new__(cls, input_array, label=None, metadata=None):
        if input_array is None:
            return None
        obj = np.asarray(input_array).view(cls)
        obj.label = label
        obj.metadata = metadata
        return obj


class STORMTable(np.ndarray):
    """STORM data array
    
    Args:
        input_array: 
    """

    def __new__(cls, input_array, label=None, metadata=None):

        if input_array is None:
            return None
        obj = np.asarray(input_array).view(cls)
        obj.label = label
        obj.metadata = metadata
        return obj


class STORMImage(np.ndarray):
    def __new__(cls, input_array, label=None, metadata=None):
        """STORM recontructed image
        
        Args:
            input_array: STORM data array 
        """
        if input_array is None:
            return None

        obj = np.asarray(input_array).view(cls)
        obj.label = label
        obj.metadata = metadata
        return obj

#todo this shoud be a dict?
class MetaData(np.ndarray):
    def __new__(cls, input_array, label=None, metadata=None):
        if input_array is None:
            return None
        obj = np.asarray(input_array).view(cls)
        obj.label = label
        obj.metadata = metadata
        return obj


class Data(object):
    """
    Parent object for all data classes
    """

    theta = 0

    #todo read metadata from config
    metadata = {
        'img_pixelsize': 80,
        'storm_pixelsize': 16,
        'shape': None
    }

    def __init__(self, binary_img=None, brightfield_img=None, fl_data=None, storm_data=None, *args, **kwargs):
        d = {'binary': binary_img, 'brightfield': brightfield_img, 'fluorescence': fl_data, 'storm': storm_data}
        data_dict = {k: v for k, v in d.items() if v is not None}

        if rotate:
            if len(data_dict) == 1:
                k, v = data_dict.items()[0]
                self.theta = self._calc_orientation(k, v)
            else:
                if type(rotate) == bool:
                    raise ValueError('Please specify from which data source to orient the cell')
                else:
                    try:
                        orient_img = data_dict[rotate]
                        self.theta = self._calc_orientation(orient_img)
                    except KeyError:
                        raise ValueError('Invalid rotation data source specified')

        if binary_img:
            if self.theta:
                binary_img = scipy_rotate(binary_img, -self.theta)
        self.binary_img = BinaryImage(binary_img)

        #if fl_data:
        if type(fl_data) == dict:
            for k, v in fl_data.items():
                raise NotImplementedError('fl data dict with multiple channels not implemented')
                #todo add to img_data

        elif type(fl_data) == np.ndarray:
            if fl_data.ndim == 2:
                if self.theta:
                    fl_data = scipy_rotate(fl_data, -self.theta)
                self.fl_img = FluorescenceImage(fl_data)
            elif fl_data.ndim == 3:
                self.fl_movie = FluorescenceMovie(fl_data)
        elif type(fl_data) == FluorescenceImage:
            self.fl_img = fl_data
        else:
            self.fl_img = None

        if type(fl_data) == FluorescenceMovie:
            self.fl_movie = fl_data
        else:
            self.fl_movie = None

        if brightfield_img:
            if self.theta:
                brightfield_img = scipy_rotate(brightfield_img, -self.theta)
        self.bf_img = BrightFieldImage(brightfield_img)

        img_data = [self.binary_img, self.bf_img, self.fl_img, self.fl_movie] # todo perhaps fl_img and movie should be unified
        shapes = [img.shape[:2] for img in img_data if img]

        assert(shapes[1:] == shapes[:-1])
        if shapes:
            self.metadata['shape'] = shapes[0]

        #todo allow multiple channels
        self.storm_data = STORMTable(storm_data)

        if storm_data is not None:
            if self.metadata['shape']:
                xmax = self.metadata['shape'][0] * self.metadata['img_pixelsize']
                ymax = self.metadata['shape'][1] * self.metadata['img_pixelsize']
            else:
                xmax = int(storm_data['x'].max()) + 2 * self.metadata['storm_pixelsize']
                ymax = int(storm_data['y'].max()) + 2 * self.metadata['storm_pixelsize']

            x_bins = np.arange(0, xmax, self.metadata['storm_pixelsize'])
            y_bins = np.arange(0, ymax, self.metadata['storm_pixelsize'])

            h, xedges, yedges = np.histogram2d(storm_data['x'], storm_data['y'], bins=[x_bins, y_bins])

            self.storm_img = STORMImage(h.T)


        #theta = self._calc_orientation(self.storm_img)

    def _rotate_storm(self, storm_data, theta):
        theta *= np.pi/180
        x = storm_data['x']
        y = storm_data['y']

        if self.metadata['shape']:
            xmax = self.metadata['shape'][0] * self.metadata['img_pixelsize']
            ymax = self.metadata['shape'][1] * self.metadata['img_pixelsize']
        else:
            xmax = int(storm_data['x'].max()) + 2 * self.metadata['storm_pixelsize']
            ymax = int(storm_data['y'].max()) + 2 * self.metadata['storm_pixelsize']

        x -= xmax / 2
        y -= ymax / 2

        xr = x*np.cos(theta) + y*np.sin(theta)
        yr = y*np.cos(theta) - x*np.sin(theta)

        xr += xmax / 2
        yr += ymax / 2

        storm_out = np.copy(storm_data)
        storm_out['x'] = xr
        storm_out['y'] = yr

        return storm_out

    def _calc_orientation(self, dtype, data):
        if dtype in ['binary', 'brightfield']:
            img = data
        elif dtype == 'storm':
            xmax = int(data['x'].max()) + 2 * self.metadata['storm_pixelsize']
            ymax = int(data['y'].max()) + 2 * self.metadata['storm_pixelsize']
            x_bins = np.arange(0, xmax, self.metadata['storm_pixelsize'])
            y_bins = np.arange(0, ymax, self.metadata['storm_pixelsize'])

            img, xedges, yedges = np.histogram2d(data['x'], data['y'], bins=[x_bins, y_bins])

        #todo multichannel support
        elif dtype == 'fl_data':
            if data.ndim == 2:
                img = data
            elif data.ndim == 3:
                img = data[0]

        com = mh.center_of_mass(img)

        mu00 = mh.moments(img, 0, 0, com)
        mu11 = mh.moments(img, 1, 1, com)
        mu20 = mh.moments(img, 2, 0, com)
        mu02 = mh.moments(img, 0, 2, com)

        mup_20 = mu20 / mu00
        mup_02 = mu02 / mu00
        mup_11 = mu11 / mu00

        theta_rad = 0.5 * math.atan(2 * mup_11 / (mup_20 - mup_02)) #todo math -> numpy
        theta = theta_rad * (180 / math.pi)
        if (mup_20 - mup_02) > 0:
            theta += 90

        return theta