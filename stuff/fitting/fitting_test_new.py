import numpy as np
import matplotlib.pyplot as plt
from colicoords.models import RDistModel, PSF
from colicoords import Data, Cell, CellPlot
from colicoords.fileIO import load, save
from colicoords.optimizers import CellFitting
import tifffile
from scipy.optimize import minimize_scalar
import os
import pickle
import time
from scipy.integrate import quad


data_dir = r'C:\Users\Smit\_processed_data\2017\20170720_NHS_control_lacySR\controls'
cell_list = load(os.path.join(data_dir, 'lacy_50.cc'))

c = cell_list[5]
cp = CellPlot(c)

x, y = c.r_dist(25, 1)
y = y / (2*y.sum())
#plt.plot(x, y)
#plt.plot(x, y, label='data')

def integrant(x, v, sigma, a1, a2, r1, r2):
    a1 /= 0.5*np.pi*r1**2
    a2 /= np.pi*r2


    res = ((1/sigma*np.sqrt(2*np.pi)) * np.exp(-((x -v)/sigma)**2 / 2)) * \
    (a1*np.nan_to_num(np.sqrt(r1**2 - x**2)) + a2*np.nan_to_num(np.sqrt(1 + (x ** 2 / (r2 ** 2 - x ** 2)))))

    return res / (2*np.pi)


sigma = 180/80
a1, a2, = 0.0, 1
r1, r2 = 4.8, 5.5

xarr = np.arange(0, 25, 0.25)

y, abserr = quad(integrant, -np.inf, np.inf, args=(2, sigma, a1, a2, r1, r2))

yarr, yerr = np.array([quad(integrant, -np.inf, np.inf, args=(v, sigma, a1, a2, r1, r2)) for v in xarr]).T

print(yarr)

print('wat')
#plt.plot(xarr, yarr)
# #plt.show()
#
for s in [140, 160, 180, 200, 220]:
    sigma = s / 80
    a1, a2, = 0.0, 1
    r1, r2 = 4.8, 384 / 80

    yarr, yerr = np.array([quad(integrant, -np.inf, np.inf, args=(v, sigma, a1, a2, r1, r2)) for v in xarr]).T
    yarr /= yarr.max()

    plt.plot(xarr, yarr, label='sigma = {}'.format(s))


plt.legend()
plt.show()
#
#


# for nump in [151, 251, 351, 361, 371]:
#
#
#     psf = PSF(sigma=160/80)
#     rmodel = RDistModel(psf, 40, num_points=nump)
#
# #rmodel.sub_par({'a1': 0.1, 'r2': 5.123921464231603, 'a2': 0.9808603524281131, 'r1': 5.123921464231603})
#
#
#     #todo when r close to x step then problem with convolution
#     #todo numerically integrate convolution and then only request required point!
#     #todo first add then convolve!
#     r = 4.80001
#     d = {'a1':0, 'a2':1, 'r2': r}
#     ym = rmodel(x, **d)
#
#     i =np.where(np.abs(rmodel.x - r) == np.abs(rmodel.x - r).min())[0]
#     print(rmodel.x[i])
#
#     plt.plot(x, ym, label='numpoints: {}, x: {}'.format(nump, rmodel.x[i]))
# plt.legend()
# #plt.show()
# plt.title('simulation r = {}'.format(r))
# plt.savefig('stepsize dependence.png')
#
#
# fit = CellFitting(rmodel, x, y)
# T = 0.0001
# #
# res, v = fit.execute('a1 a2 r1 r2')
# print(res, v)
# #
# #
# # res, v = fit.fit_parameters('r1 r2 a1 a2', bounds=True, constraint=True, basin_hop=False,)
# # print(res, v)
# # rmodel.sub_par(res)
# # print(rmodel.r2.value)
# #
# #
# # t0 = time.time()
# # res, v = fit.fit_parameters('a1 a2 r1 r2', bounds=True, constraint=True, basin_hop=True, T=T)
# # t1 = time.time() - t0
# #
# # print(t1, res, v, T)
# #
# #
# #
# # # res_list = []
# # # v_list = []
# # # #
# # # for i, c in enumerate(cell_list):
# # #     print(i)
# # #     x, y = c.r_dist(25, 1)
# # #     y = y / y.max()
# # #
# # #     fit = CellFitting(rmodel, x, y)
# # #     res, v = fit.fit_parameters('a1 a2 r1 r2', bounds=True, constraint=True, basin_hop=True, T=100)
# # #     res_list.append(res)
# # #     v_list.append(v)
# # #
# # # with open('results.pick', 'wb') as f:
# # #     pickle.dump(res_list, f)
# #
# # # with open('results_T0p1.pick', 'wb') as f:
# # #     pickle.dump(res_list, f)
# # #
# # # with open('val.pick', 'wb') as f:
# # #     pickle.dump(v_list, f)
# #
# # # with open('results.pick', 'rb') as f:
# # #     res_list = pickle.load(f)
# # #
# # # a1 = [d['a1'] for d in res_list]
# # # a2 = [d['a2'] for d in res_list]
# # # r1 = [d['r1'] for d in res_list]
# # # r2 = [d['r2'] for d in res_list]
# # #
# # # with open('results_T0p1.pick', 'rb') as f:
# # #     res_list1 = pickle.load(f)
# # #
# # #
# # # a11 = [d['a1'] for d in res_list1]
# # # a21 = [d['a2'] for d in res_list1]
# # # r11 = [d['r1'] for d in res_list1]
# # # r21 = [d['r2'] for d in res_list1]
# # #
# # # # plt.hist(a1)
# # # plt.hist(a11)
# # plt.show()
# # #
# # # plt.hist(a2)
# # # plt.show()
# # #
# # # plt.hist(r1)
# # # plt.show()
# # #
# # plt.hist(r2)
# # plt.show()
# #
# # b = np.where(np.array(a1) > 0.8)[0]
# # print(b)

# clb = cell_list[b]
#
# x, y = clb[0].r_dist(25, 1)
# plt.plot(x, y)
# plt.show()