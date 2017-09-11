import tifffile
import matplotlib.pyplot as plt
import numpy as np
from cellcoordinates.preprocess import process_cell
from cellcoordinates.plot import CellPlot

fl_img = tifffile.imread('test_data/flu1.tif')
binary_img = tifffile.imread('test_data/binary1.tif')


cell = process_cell(rotate='binary', binary_img=binary_img, flu_data={'514': fl_img})
#cell.optimize(method='binary')

plt.imshow(cell.data.binary_img)
plt.show()


x, y = np.arange(10)**2, np.arange(10)+20
print(x, y)
xm, ym = cell.coords.transform(x, y, src='cart', tgt='matrix')
print(xm, ym)
xc, yc = cell.coords.transform(xm, ym, src='matrix', tgt='cart')
print(xc, yc)

p = CellPlot(cell)
print(cell.data.data_dict.keys())
plt.imshow(cell.data.data_dict['514'])
p.plot_outline()
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