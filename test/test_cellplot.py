import tifffile
from colicoords import Data, Cell
from colicoords.plot import CellPlot
import matplotlib.pyplot as plt


binary_img = tifffile.imread('test_data/binary_1.tif')
fluorescence_img = tifffile.imread('test_data/fluorescence_1.tif')

data = Data()
data.add_data(binary_img, 'binary')
data.add_data(fluorescence_img, 'fluorescence', name='flu_514')

cell_obj = Cell(data)
cell_obj.optimize()


cp = CellPlot(cell_obj)

fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2)


cp.plot_binary_img(ax=ax0)
cp.plot_outline(ax=ax0)

ax1.imshow(cell_obj.data.data_dict['flu_514'], cmap='viridis', interpolation='nearest')
cp.plot_outline(ax=ax1)
cp.plot_midline(ax=ax1)
ax1.set_xlim(0, 55)
ax1.set_ylim(0, 55)

cp.plot_dist(ax=ax2)
cp.plot_dist(ax=ax3, norm_x=True, norm_y=True)

plt.tight_layout()
plt.show()
