from test.test_functions import generate_stormdata
import matplotlib.pyplot as plt
from colicoords.preprocess import data_to_cells
from colicoords.plot import CellPlot
from colicoords.optimizers import STORMOptimizer

data = generate_stormdata()
cell_list = data_to_cells(data, rotate='binary')
cell_list.optimize()

c = cell_list[2]
cp = CellPlot(c)

x = c.data.data_dict['storm']['x']
y = c.data.data_dict['storm']['y']

fig, axes = plt.subplots(2,2)

axes[0, 0].imshow(c.data.binary_img, interpolation='nearest')
axes[0, 0].plot(x, y, linestyle='None', marker='.', color='r', alpha=0.2)
cp.plot_outline(ax=axes[0, 0])

cp.plot_dist(ax=axes[0, 1], src='storm', storm_weights='points')

so = STORMOptimizer(c, method='photons')
so.optimize_r()
so.optimize_endcaps()
so.optimize_fit()
so.optimize_overall()

axes[1, 0].imshow(c.data.binary_img, interpolation='nearest')
axes[1, 0].plot(x, y, linestyle='None', marker='.', color='r', alpha=0.2)
cp.plot_outline(ax=axes[1, 0])

cp.plot_dist(ax=axes[1, 1], src='storm', storm_weights='points')

plt.tight_layout()
plt.show()