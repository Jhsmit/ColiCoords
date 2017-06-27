import tifffile
import matplotlib.pyplot as plt

from cellcoordinates.preprocess import process_cell

fl_img = tifffile.imread('test_data/flu1.tif')
binary_img = tifffile.imread('test_data/binary1.tif')

# plt.imshow(fl_img)
# plt.show()
#
# plt.imshow(binary_img)
# plt.show()

cell = process_cell(rotate='binary', binary_img=binary_img, fl_data={'514':fl_img})
print(cell.coords.xl, cell.coords.xr)
plt.imshow(cell.coords.rc)
plt.show()
# plt.imshow(cell.data.fl_img_514)
# plt.show()
print(cell.coords.coeff)
cell.optimize(method='binary')
print(cell.coords.coeff)


plt.imshow(cell.coords.rc)
plt.show()

plt.plot(*cell.radial_distribution(50, 1, '514'))
plt.show()