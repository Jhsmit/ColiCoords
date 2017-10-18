import tifffile
import matplotlib.pyplot as plt
import numpy as np
from colicoords.preprocess import data_to_cells
from test_functions import generate_testdata
from colicoords.plot import CellPlot


data = generate_testdata('ds3')
cell_list = data_to_cells(data, pad_width=2, cell_frac=0.5, rotate='binary')
cell_list.optimize(verbose=False)
cell = cell_list[7]
cell.optimize(verbose=True)

print(len(cell_list))
print(cell_list.length)


#tifffile.imsave('binary_1.tif', cell.data.binary_img)
#tifffile.imsave('fluorescence_1.tif', cell.data.flu_fluorescence)

cp = CellPlot(cell)

plt.figure()
#plt.imshow(cell.data.data_dict['fluorescence'], cmap='viridis')
plt.imshow(cell.data.binary_img, interpolation='nearest')
cp.plot_outline(coords='mpl')
plt.show()


#
# save('cell.tif', cell)
#
# plt.imshow(cell.data.data_dict['514'])
# plt.show()

#
# print(cell.data.data_dict['514'].dtype)
#
# with tifffile.TiffWriter('testtif.tif', imagej=True) as t:
#     t.save(cell.data.data_dict['514'])

# plt.imshow(cell.coords.y_coords)
# plt.show()
#
# plt.imshow(cell.coords.rc)
# plt.show()
#
# plt.imshow(cell.coords.xc)
# plt.show()
#
# plt.imshow(cell.coords.psi, interpolation='nearest', cmap='viridis')
# plt.show()


#
# plt.plot(*cell.radial_distribution(50, 1, '514'))
# plt.show()