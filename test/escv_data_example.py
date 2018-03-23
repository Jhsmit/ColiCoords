from colicoords import Cell, CellList, data_to_cells, Data, CellPlot
from colicoords.fileIO import load_thunderstorm
import matplotlib.pyplot as plt
import tifffile
import seaborn as sns
import numpy as np


#this assumes the pixelsize of your image is 80 nm
storm_table = load_thunderstorm('test_data/ds5/storm_table.csv')
binary = tifffile.imread('test_data/ds5/binary.tif')
fluorescence = tifffile.imread('test_data/ds5/flu.tif')

data = Data()
data.add_data(binary, 'binary')  # add binary first
data.add_data(storm_table, 'storm')
data.add_data(fluorescence, 'fluorescence')

cell_list = data_to_cells(data)
cell_list.optimize()


c = cell_list[1]
cp = CellPlot(c)


fig, axes = plt.subplots(2, 2)
x, y = c.data.data_dict['storm']['x'], c.data.data_dict['storm']['y']
axes[0, 0].plot(x, y, 'r.')
cp.plot_binary_img(ax=axes[0, 0])
cp.plot_outline(ax=axes[0, 0])

cp.imshow(c.data.data_dict['fluorescence'], ax=axes[0, 1])
cp.plot_outline(ax=axes[0, 1])
axes[0, 1].plot(x, y, 'r.')

cp.plot_dist(ax=axes[1, 0], mode='r', data_name='storm', stop=10, step=0.2)

plt.show()

#Get the number of spots per cell
n_spots = np.array([c.data.data_dict['storm'].size for c in cell_list])

plt.hist(n_spots, bins='fd')
plt.show()









# s = storm_table[storm_table['frame'] == 1]
# x, y = s['x'], s['y']
#
# plt.figure()
# plt.imshow(fluorescence[0], cmap='viridis', interpolation='none', extent=[0, 512, 512, 0])
# plt.plot(x, y, 'r.')
# plt.show()
