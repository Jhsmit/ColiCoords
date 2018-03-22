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

#c.coords.coeff[2] = -0.02

cp = CellPlot(c)
plt.figure()
cp.plot_binary_img(alpha=0.8)
cp.plot_storm('storm', kernel='gaus', alpha_cutoff=0.2)
#plt.plot(x, y, linestyle='None', marker='.', color='r', alpha=0.9)
cp.plot_outline()

import numpy as np

t = np.linspace(c.coords.xl, c.coords.xr, num=100)

a0, a1, a2 = c.coords.coeff
print(a0, a1, a2)
xt = t + c.coords.r * ((a1 + 2*a2*t) / np.sqrt(1 + (a1 + 2*a2*t)**2))
yt = a0 + a1*t + a2*(t**2) - c.coords.r * (1 / np.sqrt(1 + (a1 + 2*a2*t)**2))


a0, a1, a2 = c.coords.coeff
print(a0, a1, a2)
xt_2 = t + - c.coords.r * ((a1 + 2*a2*t) / np.sqrt(1 + (a1 + 2*a2*t)**2))
yt_2 = a0 + a1*t + a2*(t**2) + c.coords.r * (1 / np.sqrt(1 + (a1 + 2*a2*t)**2))

#
# gx = c.coords.p(x + c.coords.r*np.sin(np.arctan(c.coords.p_dx(x)))) + \
#      c.coords.r*np.cos(np.arctan(c.coords.p_dx(x)))
# cp.plot_midline()

#gp_x = c.coords.p(x) + c.coords.r*np.cos(np.arctan(c.coords.p_dx(x)))


plt.plot(xt, yt, color='b', linestyle='--')
plt.plot(xt_2, yt_2, color='b', linestyle='--')
cp._plot_intercept_line(c.coords.xl)
cp._plot_intercept_line(c.coords.xr)


cp.show()


# plt.imshow(c.data.binary_img, interpolation='nearest')
# plt.plot(x, y, linestyle='None', marker='.', color='r', alpha=0.2)
# cp.plot_outline()
# plt.show()
#
#
#
# plt.figure()
# cp.plot_dist(src='storm', storm_weights='points')
# plt.show()
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
