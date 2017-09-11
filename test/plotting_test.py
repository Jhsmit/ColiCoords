from cellcoordinates.plot import CellPlot


import tifffile
import matplotlib.pyplot as plt
import numpy as np
from cellcoordinates.preprocess import process_cell
from cellcoordinates.fileIO import save
from cellcoordinates.plot import CellPlot
from cellcoordinates.cell import Cell

fl_img = tifffile.imread('test_data/flu1.tif')
binary_img = tifffile.imread('test_data/binary1.tif')


cell = process_cell(rotate='binary', binary_img=binary_img, fl_data={'514': fl_img})
cell.optimize()

p = CellPlot(cell)


plt.figure()
p.plot_dist(mode='r')
plt.show()