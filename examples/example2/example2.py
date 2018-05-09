import tifffile
from colicoords import Cell, Data
from colicoords.preprocess import data_to_cells
from colicoords.plot import CellPlot, CellListPlot
import matplotlib.pyplot as plt

binary_stack = tifffile.imread('binary_stack_2.tif')
flu_stack = tifffile.imread('fluorescence_stack_2.tif')
brightfield_stack = tifffile.imread('brightfield_stack_2.tif')


data = Data()
data.add_data(binary_stack, 'binary')
data.add_data(flu_stack, 'fluorescence')
data.add_data(brightfield_stack, 'brightfield')

cell_list = data_to_cells(data)
cell_list.optimize(verbose=True)

clp = CellListPlot(cell_list)

fig, axes = plt.subplots(2, 2)
clp.hist_property(ax=axes[0,0], tgt='radius')
clp.hist_property(ax=axes[0,1], tgt='length')
clp.hist_property(ax=axes[1,0], tgt='area')
clp.hist_property(ax=axes[1,1], tgt='volume')
axes[0,0].set_ylim(0, 17)
plt.tight_layout()
plt.autoscale()
