import numpy as np
import matplotlib.pyplot as plt
from colicoords.models import RDistModel, PSF
from colicoords import Data, Cell
import tifffile
from scipy.optimize import minimize_scalar

binary_img = tifffile.imread('../examples/example1/binary_1.tif')
fluorescence_img = tifffile.imread('../examples/example1/fluorescence_1.tif')

data = Data()
data.add_data(binary_img, 'binary')
data.add_data(fluorescence_img, 'fluorescence', name='flu_514')

cell_obj = Cell(data)
cell_obj.optimize()

xm, r_dist = cell_obj.r_dist(40, 1)
r_dist /= r_dist.max()


psf = PSF(sigma=180/80)
rmodel = RDistModel(psf, 40)

#x, yc = rmodel.signal_cytosol(5.5)


x, y = rmodel.signal_membrane(6.7)
plt.plot(x, y)
plt.show()

def minfunc(r, rmodel, x, y):
    # x_th_m, y_th_m =
    ym = np.interp(x, *rmodel.signal_cytosol(r))
    return np.sum((y - ym)**2)


res = minimize_scalar(minfunc, args=(rmodel, xm, r_dist))

print(res.x)

x_th_m, y_th_m = rmodel.signal_cytosol(res.x)
y_th_m /= y_th_m.max()

ym = np.interp(xm, x_th_m, y_th_m)

plt.plot(xm, r_dist)
plt.plot(xm, ym)
#plt.xlim(0, 20)

plt.show()
