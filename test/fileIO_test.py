import tifffile
import matplotlib.pyplot as plt
from colicoords.fileIO import load, save

from colicoords.preprocess import process_cell

fl_img = tifffile.imread('test_data/flu1.tif')
binary_img = tifffile.imread('test_data/binary1.tif')

cell = process_cell(rotate='binary', binary_img=binary_img, flu_data={'514': fl_img})
cell.optimize(method='binary')

save('tempfile.cc', cell)

cell_load = load('tempfile.cc')