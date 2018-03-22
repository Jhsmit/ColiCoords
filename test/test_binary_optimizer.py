import matplotlib.pyplot as plt
from colicoords.preprocess import data_to_cells
from colicoords.plot import CellPlot
from colicoords.cell import Cell
from colicoords.data_models import Data
from colicoords.optimizers import Optimizer
from colicoords.config import cfg
import numpy as np
import tifffile

binary_img = tifffile.imread('test_data/binary_1.tif')
fluorescence_img = tifffile.imread('test_data/fluorescence_1.tif')

data = Data()
data.add_data(binary_img, 'binary')
data.add_data(fluorescence_img, 'fluorescence', name='flu_514')

c = Cell(data)
cp = CellPlot(c)

bo = Optimizer(c)
before = np.sum(np.logical_xor(c.data.binary_img, c.coords.rc < c.coords.r))
bo.optimize()
after = np.sum(np.logical_xor(c.data.binary_img, c.coords.rc < c.coords.r))

print('before, after', before, after)
# before, after 48 35

fix, axes = plt.subplots(2,2)
#cp.plot_binary_img(ax=axes[0, 0], alpha=0.5)
axes[0, 0].imshow(5*(c.coords.rc < c.coords.r).astype('int') + c.data.binary_img,
                  interpolation='nearest', extent=[0, c.data.shape[1], c.data.shape[0], 0], cmap='jet')
cp.plot_outline(ax=axes[0, 0])





# for k, v in res.items():
#     print(k, v)
#
# print('final value', val)
#
# cp.plot_binary_img(ax=axes[1, 0], alpha=0.5)
# cp.plot_storm('storm', ax=axes[1, 0], kernel='gaus', alpha_cutoff=0.2, upscale=4)
# cp.plot_outline(ax=axes[1, 0])
# cp.plot_midline(ax=axes[1, 0])
# axes[1,0].set_title('Optimize value {}'.format(val))
# # axes[1,0].autoscale()
#
# cp.plot_dist(ax=axes[1, 1], src='storm', storm_weights='points')
# axes[1, 1].set_xlim(0, 1.5)
# axes[1, 1].axvline(c.coords.r * (cfg.IMG_PIXELSIZE / 1000), color='r')
#
# plt.tight_layout()
#
plt.show()
# #plt.savefig('test_out/storm_example/storm_{}.png'.format(str(datetime.datetime.now())).replace(' ', '_').replace(':', '-'))