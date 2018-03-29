from colicoords import Cell, CellList, data_to_cells, CellListPlot, CellPlot
from ..test.test_functions import generate_testdata



data = generate_testdata('ds3')
cell_list = data_to_cells(data)


c = cell_list[5]
cp = CellPlot(c)

cp.plot_dist()
cp.show()