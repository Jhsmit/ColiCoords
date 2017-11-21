from test.test_functions import generate_stormdata
import matplotlib.pyplot as plt
from colicoords.preprocess import data_to_cells
from colicoords.plot import CellPlot
from colicoords.optimizers import STORMOptimizer

data = generate_stormdata()
import numpy as np
print(len(data))
for d in data:
    print('woei')

x = data.data_dict['storm']['x']
y = data.data_dict['storm']['y']


cell_list = data_to_cells(data, rotate='binary')

c = cell_list[2]
c.optimize()

x = c.data.data_dict['storm']['x']
y = c.data.data_dict['storm']['y']

cp = CellPlot(c)

plt.imshow(c.data.binary_img, interpolation='nearest')
plt.plot(x, y, linestyle='None', marker='.', color='r', alpha=0.2)
cp.plot_outline()
plt.show()



plt.figure()
cp.plot_dist(src='storm', storm_weights='points')
plt.show()

xd, rd, = c.r_dist(2, 0.1, data_name='storm', norm_x=True)

x = c.data.data_dict['storm']['x']
y = c.data.data_dict['storm']['y']

plt.imshow(c.data.binary_img, interpolation='nearest')
plt.plot(x, y, linestyle='None', marker='.', color='r', alpha=0.2)
cp.plot_outline()
plt.show()

#c.optimize(src='storm', method='points', verbose=True)

so = STORMOptimizer(c, method='photons')
print('r', c.coords.r)
so.r.value = 10
print(c.coords.r)

so.optimize_r()
so.optimize_endcaps()
so.optimize_fit()
#
# so.optimize_stepwise()
so.optimize_overall()
# print(c.area)
#
plt.imshow(c.data.binary_img, interpolation='nearest')
plt.plot(x, y, linestyle='None', marker='.', color='r', alpha=0.2)
cp.plot_outline()
plt.show()

plt.figure()
cp.plot_dist(src='storm', storm_weights='points')
plt.show()
