from test.test_functions import load_stormdata
import matplotlib.pyplot as plt
from colicoords.preprocess import data_to_cells
from colicoords.plot import CellPlot
from colicoords.optimizers import STORMOptimizer

data = load_stormdata()
cell_list = data_to_cells(data, rotate='binary')

c = cell_list[2]
c.optimize()


x = c.data.data_dict['storm']['x']
y = c.data.data_dict['storm']['y']

cp = CellPlot(c)
plt.figure()
cp.plot_binary_img(alpha=0.8)
cp.plot_storm('storm', kernel='gaus', alpha_cutoff=0.2)
plt.plot(x, y, linestyle='None', marker='.', color='r', alpha=0.9)
cp.plot_outline()

cp.show()


# plt.imshow(c.data.binary_img, interpolation='nearest')
# plt.plot(x, y, linestyle='None', marker='.', color='r', alpha=0.2)
# cp.plot_outline()
# plt.show()
#
#
#
plt.figure()
cp.plot_dist(src='storm', storm_weights='points')
plt.show()
#
# xd, rd, = c.r_dist(2, 0.1, data_name='storm', norm_x=True)
#
# x = c.data.data_dict['storm']['x']
# y = c.data.data_dict['storm']['y']
#
# plt.imshow(c.data.binary_img, interpolation='nearest')
# plt.plot(x, y, linestyle='None', marker='.', color='r', alpha=0.2)
# cp.plot_outline()
# plt.show()

#c.optimize(src='storm', method='points', verbose=True)
#
so = STORMOptimizer(c, method='photons')

so.optimize_r()
so.optimize_endcaps()
so.optimize_fit()
# #
# # so.optimize_stepwise()
# so.optimize_overall()
# # print(c.area)
# #
# plt.imshow(c.data.binary_img, interpolation='nearest')
# plt.plot(x, y, linestyle='None', marker='.', color='r', alpha=0.2)
# cp.plot_outline()
# plt.show()
#
# plt.figure()
# cp.plot_dist(src='storm', storm_weights='points')
# plt.show()
