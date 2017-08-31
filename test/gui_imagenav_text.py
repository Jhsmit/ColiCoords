from cellcoordinates.gui.controller import ImageSelectController
import sys
from PyQt4 import QtGui
import tifffile
import numpy as np
import os

def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]

bin_files = listdir_fullpath(r'test_data/ds1/binary')
bf_files = listdir_fullpath(r'test_data/ds1/brightfield')

print(bin_files)

bin_arr = np.empty((len(bin_files), 512, 512))
for i, f in enumerate(bin_files):
    bin_arr[i] = tifffile.imread(f)


bf_arr = np.empty((len(bin_files), 512, 512))
for i, f in enumerate(bf_files):
    bf_arr[i] = tifffile.imread(f)


data_dict = {}
data_dict['binary'] = bin_arr
data_dict['brightfield'] = bf_arr
print(bin_arr.shape)

app = QtGui.QApplication(sys.argv)



ctrl = ImageSelectController(data_dict, len(bin_files))



sys.exit(app.exec_())
