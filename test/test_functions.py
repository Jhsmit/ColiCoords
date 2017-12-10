from colicoords.data_models import Data
from colicoords.config import cfg
from colicoords.fileIO import load_thunderstorm
import tifffile
import numpy as np
import os


def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]

bin_files = listdir_fullpath(r'test_data/ds1/binary')
bf_files = listdir_fullpath(r'test_data/ds1/brightfield')
flu_files = listdir_fullpath(r'test_data/ds1/fluorescence')


def load_data(dataset):
    dclasses = ['binary', 'brightfield', 'fluorescence']

    data = Data()
    for dclass in dclasses:
        files = listdir_fullpath(os.path.join('test_data', dataset, dclass.lower()))
        arr = np.empty((len(files), 512, 512)).astype('uint16')
        for i, f in enumerate(files):
            arr[i] = tifffile.imread(f)
        data.add_data(arr, dclass)

    return data


def load_stormdata():
    binary = tifffile.imread(r'test_data/ds2/binary_resized.tif').astype('int')

    dtype = {
        'names': ("id", "frame", "x", "y", "sigma", "intensity", "offset", "bkgstd", "chi2", "uncertainty_xy"),
        'formats': (int, int, float, float, float, float, float, float, float, float)
    }

    storm_table = np.genfromtxt(r'test_data/ds2/storm_table.csv', skip_header=1, dtype=dtype, delimiter=',')
    storm_table['x'] /= cfg.IMG_PIXELSIZE
    storm_table['y'] /= cfg.IMG_PIXELSIZE

    data = Data()
    data.add_data(binary, 'binary')
    data.add_data(storm_table, 'storm')

    return data


def load_escvdata():
    binary = tifffile.imread(r'test_data/ds5/binary.tif')
    flu = tifffile.imread(r'test_data/ds5/flu.tif')
    storm = load_thunderstorm(r'test_data/ds5/storm_table.csv')

    data = Data()
    data.add_data(binary, 'binary')
    data.add_data(flu, 'fluorescence')
    data.add_data(storm, 'storm')

    return data


def generate_stormdata():
    x = np.array([10, 30, 30, 40, 40.25, 30, 25, 20, 15, 10]) + 0.5
    y = np.array([25, 25, 10, 30, 30, 30, 30, 30, 30, 30]) + 0.5
    frame = np.ones(10)

    storm_data = np.zeros((10, ), dtype=[('frame', int), ('x', float), ('y', float)])
    storm_data['x'] = x
    storm_data['y'] = y
    storm_data['frame'] = frame

    binary_img = tifffile.imread('test_data/ds6/binary_1.tif')

    data = Data()
    data.add_data(binary_img, 'binary')
    data.add_data(storm_data, 'storm')

    return data


#todo refactor to load
def generate_testdata(dataset):
    if dataset == 'ds2':
        return load_stormdata()
    elif dataset in ['ds1', 'ds3', 'ds4']:
        return load_data(dataset)
    elif dataset is 'ds5':
        return load_escvdata()
    elif dataset == 'ds6':
        return generate_stormdata()


if __name__ == '__main__':
    data = generate_testdata('ds4')
    tifffile.imsave('binary_stack_2.tif', data.binary_img)
    tifffile.imsave('brightfield_stack_2.tif', data.brightfield_img)
    tifffile.imsave('fluorescence_stack_2.tif', data.data_dict['fluorescence'])