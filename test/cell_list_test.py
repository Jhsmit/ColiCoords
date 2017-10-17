from test_functions import generate_testdata
from colicoords.gui.controller import CellObjectController
from colicoords.cell import CellList
from colicoords.plot import CellPlot, CellListPlot

import seaborn as sns
import matplotlib.pyplot as plt


data = generate_testdata()
ctrl = CellObjectController(data, '')
cell_list = ctrl._create_cell_objects(data, 0.5, 2, 'Binary')
print(cell_list)

cl = CellList(cell_list[:10])
cl.optimize()

c = cl[0]
p = CellPlot(c)


clp = CellListPlot(cl)
# plt.figure()
# p.plot_outline(coords='mpl')
# plt.imshow(c.data.binary_img)
# plt.show()



# clp.hist_property('length')
# plt.show()
#
# plt.figure()
# clp.hist_property('radius')
# plt.show()
#
# plt.figure()
# clp.hist_property('area')
# plt.show()
#
# plt.figure()
# clp.hist_property('volume')
# plt.show()

plt.figure()
clp.plot_dist(mode='r', src='Fluorescence', norm_y=False)
plt.show()