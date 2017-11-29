from test.test_functions import generate_testdata
import matplotlib.pyplot as plt
from colicoords.preprocess import data_to_cells
from colicoords.plot import CellPlot


data = generate_testdata('ds5')

print(data.shape)

print(data.storm_storm['x'])


ss = data[1:3, :, :]
print(ss.shape)

print(data.storm_storm['x'])
print(data.storm_storm['frame'])