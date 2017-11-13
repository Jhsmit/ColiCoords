from test.test_functions import generate_testdata
import matplotlib.pyplot as plt
from colicoords.preprocess import data_to_cells
from colicoords.plot import CellPlot


data = generate_testdata('ds5')

print(data.shape)
#
# print(data.storm_storm['x'])
# print(data.storm_storm['frame'])
print(data.storm_storm.size)
#
#
# ss = data[2]
# #
# print(ss.shape)
#
# print(ss.storm_storm['x'])
# print(ss.storm_storm['frame'])


for ds in data:
    print('size', ds.storm_storm.size)


#
# cell_list = data_to_cells(data)
#
# for c in cell_list:
#     print(c.data.storm_storm.size)