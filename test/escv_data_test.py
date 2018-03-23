from test.test_functions import generate_testdata
import matplotlib.pyplot as plt
from colicoords.preprocess import data_to_cells
from colicoords.plot import CellPlot
from colicoords import Cell

import numpy as np

data = generate_testdata('ds5')



data[5:60, 10:100]

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


for i, d in enumerate(data):
    print('size d', d.data_dict['storm'].size)
#
#
# x = ss.data_dict['storm']['x']
# y = ss.data_dict['storm']['y']
#
#
# print('yvals before', y)
#
#
# cell_list = data_to_cells(ss, rotate='binary')
#
# # for c in cell_list:
# #     print(c.data.data_dict['storm']['x'])
#
#
# print(cell_list[1].data.data_dict['storm']['y'])
#
# for c in cell_list:
#     c.optimize()
#
# print(len(cell_list))
#
# c = cell_list[1]
#
#
# # plt.figure()
# # cp.plot_dist(src='storm')
# # plt.xlim(0, 2)
# # plt.show()
#
# # xd, rd, = c.r_dist(10, 1, data_name='storm')
# # plt.plot(xd, rd)
# # plt.show()
#
# x = c.data.data_dict['storm']['x']
# y = c.data.data_dict['storm']['y']
#
# print(x, y)
#
# plt.imshow(c.data.binary_img, interpolation='nearest')
# plt.plot(x, y, linestyle='None', marker='.', color='r')#, alpha=0.2)
# plt.show()
#
# plt.imshow(c.data.binary_img, interpolation='nearest')
# #cp.plot_outline()
# plt.plot(x, y, linestyle='None', marker='.', color='r')#, alpha=0.2)
# plt.show()
#
# cp = CellPlot(c)
#
# fig, axes = plt.subplots(1, 2)
# cp.plot_binary_img(ax=axes[0])
# cp.plot_storm(ax=axes[0], data_name='storm', kernel='gaus', alpha=0.9)
# axes[0].plot(x, y, linestyle='None', marker='.', color='r', alpha=0.9)
# # cp.plot_outline()
#
#
# r_data = c.data.rotate(50)
# rc = Cell(r_data)
#
#
# rcp = CellPlot(rc)
#
# rx = rc.data.data_dict['storm']['x']
# ry = rc.data.data_dict['storm']['y']
#
# rcp.plot_binary_img(ax=axes[1])
# rcp.plot_storm(ax=axes[1], data_name='storm', kernel='gaus', alpha=0.9)
# axes[1].plot(rx, ry, linestyle='None', marker='.', color='r', alpha=0.9)
# # cp.plot_outline()
#
# plt.show()