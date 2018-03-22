from colicoords.fileIO import load, save
from colicoords import CellList, Cell

cell_list = load('Neomycin.cc')

empty_cell_list = CellList([Cell(c.data, name=str(i)) for i, c in enumerate(cell_list)])
save('preoptimized.cc', empty_cell_list)
save('preoptimized_10.cc', CellList(empty_cell_list[:10]))
save('preoptimized_200.cc', CellList(empty_cell_list[:200]))
save('preoptimized_1000.cc', CellList(empty_cell_list[:1000]))

assert len(load('preoptimized.cc')) == len(cell_list)
assert len(load('preoptimized_10.cc')) == 10
assert len(load('preoptimized_200.cc')) == 200
assert len(load('preoptimized_1000.cc')) == 1000
