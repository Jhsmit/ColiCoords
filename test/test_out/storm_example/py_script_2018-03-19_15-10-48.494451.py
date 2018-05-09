from test.test_functions import load_stormdata
import matplotlib.pyplot as plt
from colicoords.preprocess import data_to_cells
from colicoords.plot import CellPlot
from colicoords.optimizers import Optimizer
from colicoords.config import cfg
import datetime
#tempshizzle
import numpy as np
import shutil
import seaborn as sns

data = load_stormdata()
cell_list = data_to_cells(data, rotate='binary')

c = cell_list[2]
c.optimize()
cp = CellPlot(c)

x = c.data.data_dict['storm']['x']
y = c.data.data_dict['storm']['y']

fig, axes = plt.subplots(2, 2)

cp.plot_binary_img(ax=axes[0, 0], alpha=0.5)
#todo gaus aanpassen
cp.plot_storm('storm', ax=axes[0, 0], kernel='gaus', alpha_cutoff=0.2, upscale=4)

#axes[0, 0].plot(x, y, linestyle='None', marker='.', color='g', alpha=0.2)
cp.plot_outline(ax=axes[0, 0])

cp.plot_dist(ax=axes[0, 1], data_name='storm', storm_weights='points')
axes[0, 1].set_xlim(0, 1.5)
axes[0, 1].axvline(c.coords.r * (cfg.IMG_PIXELSIZE / 1000), color='r')

so = Optimizer(c, data_name='storm', objective='leastsq')


res, val = so.optimize_stepwise(bounds=True, obj_kwargs={'r_lower': lambda x: 0.5*np.std(x)})

print('hoidoei')
# so.optimize_parameters('r')
# so.optimize_parameters('xl xr')
# so.optimize_parameters('a0 a1 a2')
# res, val = so.optimize_parameters('r xl xr a0 a1 a2', minimize_func='leastsq', bounds=True)
#res, val = so.optimize_parameters('r xl xr a0 a1 a2', minimize_func='leastsq')

for k, v in res.items():
    print(k, v)

print('final value', val)

cp.plot_binary_img(ax=axes[1, 0], alpha=0.5)

storm_table = c.data.data_dict['storm']
r = c.coords.calc_rc(storm_table['x'], storm_table['y'])
b = r > r.mean() - 1*r.std()
selected_table = storm_table[b]

cp._plot_storm(selected_table, ax=axes[1, 0], kernel='gaus', alpha_cutoff=0.2, upscale=4)

b = r < r.mean() - 1*r.std()
drop_table = storm_table[b]
cp._plot_storm(drop_table, ax=axes[1, 0], kernel='gaus', alpha_cutoff=0.2, upscale=4, cmap=sns.light_palette("blue", as_cmap=True))


cp.plot_outline(ax=axes[1, 0])
cp.plot_midline(ax=axes[1, 0])
axes[1,0].set_title('Optimize value {}'.format(val))

cp.plot_dist(ax=axes[1, 1], data_name='storm', storm_weights='points')
axes[1, 1].set_xlim(0, 1.5)
axes[1, 1].axvline(c.coords.r * (cfg.IMG_PIXELSIZE / 1000), color='r')

plt.tight_layout()

#plt.show()
plt.savefig('test_out/storm_example/storm_{}.png'.format(str(datetime.datetime.now())).replace(' ', '_').replace(':', '-'))


shutil.copy('storm_test_plot.py', 'test_out/storm_example/py_script_{}.py'.format(str(datetime.datetime.now())).replace(' ', '_').replace(':', '-'))