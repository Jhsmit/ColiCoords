import os
from colicoords.fileIO import load, save
import matplotlib.pyplot as plt
from colicoords.preprocess import batch_flu_images


def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]

binary_files = [os.path.join(os.getcwd(), 'test_data/ds1/binary', f) for f in os.listdir('test_data/ds1/binary')]
print(binary_files)
bf_files = [os.path.join(os.getcwd(), 'test_data/ds1/brightfield', f) for f in os.listdir('test_data/ds1/brightfield')]
fl_files = [os.path.join(os.getcwd(), 'test_data/ds1/fluorescence', f) for f in os.listdir('test_data/ds1/fluorescence')]

fl_dict = {'514': fl_files}

cell_list = list([c for c in batch_flu_images(binary_files, fl_dict, bf_files=bf_files)])

print(len(cell_list))
print(cell_list[0].coords.r)
for c in cell_list:
    c.optimize(verbose=False)

print(cell_list[0].coords.r)

