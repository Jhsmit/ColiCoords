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
        assert input_array.dtype == 'int'
        #todo test connectedness l, n = mh.label(bool_arr)

        #todo binary is now saved as int

        obj = np.asarray(input_array).view(cls)
        obj.label = label
        obj.metadata = metadata
        return obj


class BrightFieldImage(np.ndarray):
    def __new__(cls, input_array, name=None, metadata=None):
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

#todo this shoud be a dict? (use open microscopy format?) (XML)
class MetaData(dict):
    pass


class Data(object):
    """
    Parent object for all data classes
    """
    #todo move this from init to function calls so empty data class can be initiated
    def __init__(self, binary_img=None, bf_img=None, fl_data=None, storm_data=None, *args, **kwargs):
        img_data = [binary_img, bf_img] + [v for v in fl_data.values()]
        shapes = [img.shape[:2] for img in img_data if img is not None]
        assert (shapes[1:] == shapes[:-1])
        self.shape = shapes[0] if len(shapes) > 0 else None

        self.binary_img = BinaryImage(binary_img)
        self.bf_img = BrightFieldImage(bf_img)

        self.fl_dict = {}
        for k, v in fl_data.items():
                d = FluorescenceImage(v)
                self.fl_dict[k] = d
                setattr(self, 'flu_' + k, d)

        self.storm_data = STORMTable(storm_data)

        if storm_data is not None:
            if self.shape:
                xmax = self.shape[0] * cfg.IMG_PIXELSIZE
                ymax = self.shape[1] * cfg.IMG_PIXELSIZE
            else:
                xmax = int(storm_data['x'].max()) + 2 * cfg.STORM_PIXELSIZE
                ymax = int(storm_data['y'].max()) + 2 * cfg.STORM_PIXELSIZE

            x_bins = np.arange(0, xmax, cfg.STORM_PIXELSIZE)
            y_bins = np.arange(0, ymax, cfg.STORM_PIXELSIZE)

            h, xedges, yedges = np.histogram2d(storm_data['x'], storm_data['y'], bins=[x_bins, y_bins])

            self.storm_img = STORMImage(h.T)

        self.data_dict = {'binary': self.binary_img,
                          'brightfield': self.bf_img,
                          'storm_data': self.storm_data}
        self.data_dict.update(self.fl_dict)