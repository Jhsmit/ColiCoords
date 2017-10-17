from PyQt4 import QtGui
import sys
from colicoords.gui.controller import CellObjectController
from colicoords.data_models import Data
import tifffile
import os
import numpy as np


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
data.add_data(bin_arr, 'Binary')
data.add_data(bf_arr, 'Brightfield')
data.add_data(flu_arr, 'Fluorescence')
print(data.brightfield_img.shape)

data_s = data[:2, :, :]
print(data_s.shape)
app = QtGui.QApplication(sys.argv)

ctrl = CellObjectController(data_s, 'test_data/output')
ctrl.show()

sys.exit(app.exec_())