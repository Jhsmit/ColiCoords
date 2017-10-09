from cellcoordinates.data_models import Data
import tifffile
import numpy as np
import os

def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]

bin_files = listdir_fullpath(r'test_data/ds1/binary')
bf_files = listdir_fullpath(r'test_data/ds1/brightfield')
flu_files = listdir_fullpath(r'test_data/ds1/fluorescence')


def generate_testdata():

    bin_arr = np.empty((len(bin_files), 512, 512)).astype('uint16')
    for i, f in enumerate(bin_files):
        bin_arr[i] = tifffile.imread(f)

    bf_arr = np.empty((len(bin_files), 512, 512)).astype('uint16')
    for i, f in enumerate(bf_files):
        bf_arr[i] = tifffile.imread(f)

    flu_arr = np.empty((len(bin_files), 512, 512)).astype('uint16')
    for i, f in enumerate(flu_files):
        flu_arr[i] = tifffile.imread(f)

    data = Data()
    data.add_data(bin_arr, 'Binary')
    data.add_data(bf_arr, 'Brightfield')
    data.add_data(flu_arr, 'Fluorescence')

    return data


def generate_stormdata():
    binary = tifffile.imread(r'test_data/ds2/binary_resized.tif').astype('int')

    dtype = {
        'names': ("id", "frame", "x", "y", "sigma", "intensity", "offset", "bkgstd", "chi2", "uncertainty_xy"),
        'formats': (int, int, float, float, float, float, float, float, float, float)
    }

    storm_table = np.genfromtxt(r'test_data/ds2/storm_table.csv', skip_header=1, dtype=dtype, delimiter=',')

    data = Data()
    data.add_data(binary, 'Binary')
    data.add_data(storm_table, 'STORMTable')

    return data