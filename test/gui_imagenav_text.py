from cellcoordinates.gui.controller import ImageSelectController
from cellcoordinates.data import Data
import sys
from PyQt4 import QtGui
import tifffile
import numpy as np
import os

def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]

bin_files = listdir_fullpath(r'test_data/ds1/binary')
bf_files = listdir_fullpath(r'test_data/ds1/brightfield')
flu_files = listdir_fullpath(r'test_data/ds1/fluorescence')


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
data.add_data(bin_arr, 'binary')
data.add_data(bf_arr, 'brightfield')
print(data.brightfield_img.shape)
#data.add_data()

app = QtGui.QApplication(sys.argv)



ctrl = ImageSelectController(data, len(bin_files))



sys.exit(app.exec_())
