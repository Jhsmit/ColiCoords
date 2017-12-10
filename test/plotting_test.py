from colicoords.plot import CellPlot


import tifffile
import matplotlib.pyplot as plt
import numpy as np
from colicoords.preprocess import data_to_cells
from colicoords.fileIO import save
from colicoords.plot import CellPlot, CellListPlot
from colicoords.cell import Cell
from test_functions import generate_testdata

data = generate_testdata('ds3')
cell_list = data_to_cells(data, pad_width=2, cell_frac=0.5, rotate='binary')
#cell_list.optimize(verbose=True)
cell = cell_list[0]
cell.optimize(verbose=False)

p = CellPlot(cell)
clp = CellListPlot(cell_list)

plt.figure()
p.plot_dist(mode='r', norm_x=True)
plt.show()

plt.imshow(cell.data.data_dict['fluorescence'])
p.plot_midline()

plt.show()

plt.figure()
p.plot_dist(mode='r',)
plt.show()

plt.figure()
clp.plot_dist(mode='r', norm_y=True, norm_x=True)
plt.show()
#
plt.figure()
clp.plot_dist(mode='r', norm_y=True, norm_x=False)
plt.show()