import tifffile
from colicoords import Data, Cell
from colicoords.plot import CellPlot
import matplotlib.pyplot as plt
import numpy as np
import operator

binary_img = tifffile.imread('test_data/binary_1.tif')
fluorescence_img = tifffile.imread('test_data/fluorescence_1.tif')

data = Data()
data.add_data(binary_img, 'binary')
data.add_data(fluorescence_img, 'fluorescence', name='flu_514')

cell_obj = Cell(data)
cell_obj.optimize()
#cell_obj.coords.coeff[2] *= -50

cp = CellPlot(cell_obj)



x = cell_obj.coords.x_coords
y = cell_obj.coords.y_coords

xl = cell_obj.coords.xl
xr = cell_obj.coords.xr

xarr = np.linspace(xl - 10, xl + 10, num=200)

img = np.zeros_like(x)

op = operator.lt if cell_obj.coords.p_dx(xl) > 0 else operator.gt
img[op(y, cell_obj.coords.q(x, xl))] = 1
print(op)
#img[y < cell_obj.coords.q(x, xl)] = 1


op = operator.gt if cell_obj.coords.p_dx(xr) > 0 else operator.lt
print(op)
img[op(y, cell_obj.coords.q(x, xr))] = 2

# plt.imshow(img, interpolation='none', extent=(0, img.shape[1], img.shape[0], 0))
# cp.plot_midline()
# cp._plot_intercept_line(xl)
# cp._plot_intercept_line(xr)
# plt.plot(xarr, cell_obj.coords.q(xarr, xl))
#
# plt.show()

plt.imshow(cell_obj.coords.rc, cmap='jet')
cp._plot_intercept_line(xl)
cp._plot_intercept_line(xr)
plt.show()