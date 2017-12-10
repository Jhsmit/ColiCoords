from test.test_functions import load_stormdata
import matplotlib.pyplot as plt
from colicoords.preprocess import data_to_cells
from colicoords.plot import CellPlot
from colicoords.optimizers import STORMOptimizer
from colicoords.config import cfg
import datetime
#tempshizzle
import numpy as np

data = load_stormdata()
cell_list = data_to_cells(data, rotate='binary')

c = cell_list[2]
c.optimize()
cp = CellPlot(c)

x = c.data.data_dict['storm']['x']
y = c.data.data_dict['storm']['y']

fig, axes = plt.subplots(2, 2)

cp.plot_binary_img(ax=axes[0,0], alpha=0.5)
cp.plot_storm('storm', ax=axes[0, 0], kernel='gaus', alpha_cutoff=0.2, upscale=4)

#axes[0, 0].plot(x, y, linestyle='None', marker='.', color='g', alpha=0.2)
cp.plot_outline(ax=axes[0, 0])

cp.plot_dist(ax=axes[0, 1], src='storm', storm_weights='points')
axes[0, 1].set_xlim(0, 1.5)
axes[0, 1].axvline(c.coords.r * (cfg.IMG_PIXELSIZE / 1000), color='r')

so = STORMOptimizer(c, method='photons')
so.optimize_r()
so.optimize_endcaps()
so.optimize_fit()
so.optimize_overall()

cp.plot_binary_img(ax=axes[1, 0], alpha=0.5)
cp.plot_storm('storm', ax=axes[1, 0], kernel='gaus', alpha_cutoff=0.2, upscale=4)
cp.plot_outline(ax=axes[1, 0])

cp.plot_dist(ax=axes[1, 1], src='storm', storm_weights='points')
axes[1, 1].set_xlim(0, 1.5)
axes[1, 1].axvline(c.coords.r * (cfg.IMG_PIXELSIZE / 1000), color='r')

plt.tight_layout()

plt.savefig('test_out/storm_example/storm_{}.png'.format(str(datetime.datetime.now())).replace(' ', '_').replace(':', '-'))