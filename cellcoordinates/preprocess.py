import mahotas as mh
import numpy as np
import math
import tifffile
import os

from scipy.ndimage.interpolation import rotate as scipy_rotate
from data import Data, BinaryImage, FluorescenceImage, STORMTable, STORMImage
from cell import Cell
from config import cfg

#temp
import matplotlib.pyplot as plt


def batch_flu_images(binary_files, flu_files_dict, bf_files=None, pad_width=2, cell_frac=0.5, rotate='binary'):
    #only when fluorescence images are available
    assert type(flu_files_dict) == dict
    for a in flu_files_dict.values():
        assert len(binary_files) == len(a)

    for i, b_fp in enumerate(binary_files):
        binary = tifffile.imread(b_fp)
        b_name = os.path.splitext(os.path.basename(b_fp))[0]

        # Check the binary image for a too large fraction of cells or no cells
        if (binary > 0).mean() > cell_frac or binary.mean() == 0.:
            print('Image {}: Too many or no cells'.format(b_name))
            continue

        fl_data = {}
        for k, v in flu_files_dict.items():
            fl_data[k] = tifffile.imread(v[i])

        bf_img = tifffile.imread(bf_files[i]) if bf_files else None

        for cell in cell_generator(binary, fl_data=fl_data, bf_img=bf_img, pad_width=pad_width, rotate=rotate, img_name=b_name):
            yield cell


def cell_generator(labeled_binary, fl_data=None, bf_img=None, storm_data=None, pad_width=2, rotate=True, img_name='unknown'):
    for i in np.unique(labeled_binary)[1:]:
        binary = (labeled_binary == i).astype('int')
        min1, max1, min2, max2 = mh.bbox(binary)
        min1p, max1p, min2p, max2p = min1 - pad_width, max1 + pad_width, min2 - pad_width, max2 + pad_width
        bin_selection = labeled_binary[min1p:max1p, min2p:max2p]

        try:
            assert min1p > 0 and min2p > 0 and max1p < binary.shape[0] and max2p < binary.shape[1]
        except AssertionError:
            print('Cell {} on image {}: on the edge of the image'.format(i, img_name))
            continue

        try:
            assert len(np.unique(bin_selection)) == 2
        except AssertionError:
            print('Cell {} on image {}: multiple cells per selection'.format(i, img_name))
            continue

        bin_selection = bin_selection.astype(bool)

        fl_selection = None
        if fl_data:
            fl_selection = {}
            for k, v in fl_data.items():
                fl_selection[k] = v[min1-pad_width:max1+pad_width, min2-pad_width:max2+pad_width]

        bf_selection = bf_img[min1-pad_width:max1+pad_width, min2-pad_width:max2+pad_width] if bf_img is not None else None

        if storm_data:
            raise NotImplementedError('STORM data handling not yet implemented')

        cell = process_cell(binary_img=bin_selection, bf_img=bf_selection, fl_data=fl_selection, storm_data=None, rotate=rotate)
        yield cell


def process_cell(binary_img=None, bf_img=None, fl_data=None, storm_data=None, rotate=True):
    #binary etc images are pre-padded
    d = {'binary': binary_img, 'brightfield': bf_img, 'fluorescence': fl_data, 'storm': storm_data}
    data_dict = {k: v for k, v in d.items() if v is not None}
    assert len(data_dict) != 0
    theta = 0

    if rotate:
        #todo rotate by fluorescence
        #if there is only one data source use that one for orientation
        if len(data_dict) == 1:
            k, v = data_dict.items()[0]
            theta = _calc_orientation(k, v)
        else:
            if type(rotate) == bool:
                raise ValueError('Please specify from which data source to orient the cell')
            else:
                try:
                    orient_img = data_dict[rotate]
                    theta = _calc_orientation(rotate, orient_img)
                except KeyError:
                    raise ValueError('Invalid rotation data source specified')

    if binary_img is not None:
        binary_img = scipy_rotate(binary_img.astype('int'), -theta)

    # if fl_data:
    fl_dict = {}
    if type(fl_data) == dict:
        for k, v in fl_data.items():
            #todo: cval
            if v.ndim == 2:
                fl_dict[k] = scipy_rotate(v, -theta)
            elif v.ndim == 3:
                fl_dict[k] == scipy_rotate(v, -theta, axes=(1,0))

    if bf_img is not None:
        bf_img = scipy_rotate(bf_img, -theta)

    img_data = [binary_img, bf_img] + [v for v in fl_dict.values()]  # todo perhaps fl_img and movie should be unified
    shapes = [img.shape[:2] for img in img_data if img is not None]
    assert (shapes[1:] == shapes[:-1])
    shape = shapes[0] if len(shapes) > 0 else None

    if storm_data is not None:
        storm_data = _rotate_storm(storm_data, theta, shape=shape)

    return Cell(bf_img=bf_img, binary_img=binary_img, fl_data=fl_dict, storm_data=storm_data)


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


def _calc_orientation(dtype, data):
    if dtype in ['binary', 'brightfield']:
        img = data
    elif dtype == 'storm':
        xmax = int(data['x'].max()) + 2 * cfg.STORM_PIXELSIZE
        ymax = int(data['y'].max()) + 2 * cfg.STORM_PIXELSIZE
        x_bins = np.arange(0, xmax, cfg.STORM_PIXELSIZE)
        y_bins = np.arange(0, ymax, cfg.STORM_PIXELSIZE)

        img, xedges, yedges = np.histogram2d(data['x'], data['y'], bins=[x_bins, y_bins])

    # todo multichannel support
    elif dtype == 'fl_data':
        if data.ndim == 2:
            img = data
        elif data.ndim == 3:
            img = data[0]
    else:
        raise ValueError('Invalid dtype')

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