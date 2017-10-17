from colicoords.data_models import Data
from colicoords.config import cfg
import tifffile
import numpy as np
import os

def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]

bin_files = listdir_fullpath(r'test_data/ds1/binary')
bf_files = listdir_fullpath(r'test_data/ds1/brightfield')
flu_files = listdir_fullpath(r'test_data/ds1/fluorescence')


def generate_data(dataset):
    dclasses = ['Binary', 'Brightfield', 'Fluorescence']

    data = Data()
    for dclass in dclasses:
        files = listdir_fullpath(os.path.join('test_data', dataset, dclass.lower()))
        arr = np.empty((len(files), 512, 512)).astype('uint16')
        for i, f in enumerate(files):
            arr[i] = tifffile.imread(f)

        data.add_data(arr, dclass)

    return data


def generate_stormdata():
    binary = tifffile.imread(r'test_data/ds2/binary_resized.tif').astype('int')

    dtype = {
        'names': ("id", "frame", "x", "y", "sigma", "intensity", "offset", "bkgstd", "chi2", "uncertainty_xy"),
        'formats': (int, int, float, float, float, float, float, float, float, float)
    }

    storm_table = np.genfromtxt(r'test_data/ds2/storm_table.csv', skip_header=1, dtype=dtype, delimiter=',')
    storm_table['x'] /= cfg.IMG_PIXELSIZE
    storm_table['y'] /= cfg.IMG_PIXELSIZE

    data = Data()
    data.add_data(binary, 'Binary')
    data.add_data(storm_table, 'STORMTable')

    return data

def generate_testdata(dataset):
    if dataset == 'ds2':
        return generate_stormdata()
    elif dataset in ['ds1', 'ds3']:
        return generate_data(dataset)
