from test_functions import generate_testdata
from cellcoordinates.gui.controller import CellObjectController
from cellcoordinates.cell import CellList
from cellcoordinates.plot import CellPlot

import seaborn as sns
import matplotlib.pyplot as plt


data = generate_testdata()
ctrl = CellObjectController(data, '')
cell_list = ctrl._create_cell_objects(data, 0.5, 2, 'Binary')
print(cell_list)

cl = CellList(cell_list)

c = cl[0]
p = CellPlot(c)

print(c.coords)
print(c.label)



length = cl.length

print(len(length))
sns.distplot(length)
sns.plt.show()

plt.hist(length, bins='fd')
plt.show()