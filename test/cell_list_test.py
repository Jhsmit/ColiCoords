from test_functions import generate_testdata
from cellcoordinates.gui.controller import CellObjectController
from cellcoordinates.cell import CellList


data = generate_testdata()
ctrl = CellObjectController(data, '')
cell_list = ctrl._create_cell_objects(data, 0.5, 2, 'Binary')
print(cell_list)

cl = CellList(cell_list)

for c in cl:
    print(c)

c5 = cl[5]
print(c5)
print(len(cl))

del cl[5]

print(len(cl))

for c in reversed(cl):
    print(c)

print(cl[3] in cl)