import numpy as np
import matplotlib.pyplot as plt
from colicoords.models import RDistModel, PSF
from colicoords import Data, Cell, CellPlot
from colicoords.fileIO import load, save
from colicoords.optimizers import CellFitting

import tifffile
from scipy.optimize import minimize_scalar
import os

data_dir = r'C:\Users\Smit\_processed_data\2017\20170720_NHS_control_lacySR\controls'

cell_list = load(os.path.join(data_dir, 'lacy_50.cc'))

print(len(cell_list))

c = cell_list[0]
cp = CellPlot(c)
plt.figure()
cp.plot_dist()
plt.show()

x, y = c.r_dist(25, 1)
y = y / y.max()

plt.plot(x, y)
plt.show()

psf = PSF(sigma=180/80)
rmodel = RDistModel(psf, 40)

fit = CellFitting(rmodel, x, y)
res, v = fit.fit_parameters('a2 r2')

print(res)
print('old v', v)

rmodel.sub_par(res)

ym = res['a2']*rmodel.get_y2(x, res['r2'])
plt.plot(x, ym)
plt.plot(x, y)
plt.show()

ym = rmodel.get_y(x, res) #callable

plt.plot(x, ym)
plt.show()

#rmodel.r2.value=1
res, v = fit.fit_parameters('a1 a2 r1 r2', bounds=True)
print('new v', v)
print(res)


# ym = rmodel.get_y(x, res) #callable
#
# plt.plot(x, ym)
# plt.plot(x, y)
# plt.show()

rmodel.a1.value = 0.1
rmodel.a2.value = 1
rmodel.r1.max = 5.5
res1, v = fit.fit_parameters('a1 a2 r1 r2', bounds=True)
print('new v', v)

ym = rmodel.get_y(x, res1) #callable
print(res1)


plt.plot(x, ym)
plt.plot(x, y)
plt.show()

rmodel.a1.value = 1
rmodel.a2.value = 0.1
rmodel.r1.max = 4.5
res, v = fit.fit_parameters('a1 a2 r1 r2', bounds=True)
print('new v', v)

ym = rmodel.get_y(x, res) #callable
print(res)

rmodel.sub_par(res1)

res, v = fit.test_basinhop('a1 a2 r1 r2', bounds=True, constraint=True)
print(res)
print(v)


n_rmodel = RDistModel(psf, 40)
n_fit = CellFitting(n_rmodel, x, y)
res, val = n_fit.test_basinhop('a1 a2 r1 r2', bounds=True, constraint=True)
print('new resval', res, val)

