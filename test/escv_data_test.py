from test.test_functions import generate_testdata
import matplotlib.pyplot as plt
from colicoords.preprocess import data_to_cells
from colicoords.plot import CellPlot

import numpy as np

data = generate_testdata('ds5')

from scipy.ndimage.interpolation import rotate




#print(data.shape)
#
# print(data.storm_storm['x'])
# print(data.storm_storm['frame'])
#print(data.storm_storm.size)
#
#
# ss = data[2]
# #
# print(ss.shape)
#
# print(ss.storm_storm['x'])
# print(ss.storm_storm['frame'])

#
# for ds in data:
#     print('size', ds.storm_storm.size)

ss = data[0]

for d in ss:
    print('size', d.storm_storm.size)

# #
cell_list = data_to_cells(ss)

for c in cell_list:
    c.optimize()

cp = CellPlot(cell_list[1])
c = cell_list[1]

plt.figure()
cp.plot_dist(src='storm')
plt.show()

xd, rd, = c.r_dist(2, 0.005, src='storm')
plt.plot(xd, rd)
plt.show()

x = c.data.data_dict['storm']['x']
y = c.data.data_dict['storm']['y']

print(x, y)

#plt.imshow(c.data.binary_img, interpolation='nearest')
plt.plot(x, y, linestyle='None', marker='.', color='r')#, alpha=0.2)
plt.show()

plt.imshow(c.data.binary_img, interpolation='nearest')
cp.plot_outline()
plt.plot(x, y, linestyle='None', marker='.', color='r')#, alpha=0.2)
plt.show()