import tifffile
from colicoords import Data, Cell
from colicoords.plot import CellPlot
import matplotlib.pyplot as plt

binary_img = tifffile.imread('binary_1.tif')
fluorescence_img = tifffile.imread('fluorescence_1.tif')

data = Data()
data.add_data(binary_img, 'binary')
data.add_data(fluorescence_img, 'fluorescence', name='flu_514')

cell_obj = Cell(data)
cell_obj.optimize()

cp = CellPlot(cell_obj)

plt.figure()
plt.imshow(cell_obj.data.data_dict['flu_514'], cmap='viridis', interpolation='nearest')
cp.plot_outline()
cp.plot_midline()
plt.xlim(0, 55)
plt.ylim(0, 55)
plt.show()

f, (ax1, ax2) = plt.subplots(1, 2)
cp.plot_dist(ax=ax1)
cp.plot_dist(ax=ax2, norm_x=True, norm_y=True)
plt.tight_layout()
plt.show()