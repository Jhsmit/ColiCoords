from test.test_functions import generate_stormdata
import matplotlib.pyplot as plt
from colicoords.preprocess import data_to_cells
from colicoords.plot import CellPlot


data = generate_stormdata()
import numpy as np
print(len(data))
for d in data:
    print('woei')

# plt.imshow(data.storm_img)
# plt.show()


x = data.data_dict['storm']['x']
y = data.data_dict['storm']['y']


cell_list = data_to_cells(data, rotate='binary')
print(len(cell_list))
print(data.shape)

c = cell_list[2]
cp = CellPlot(c)

plt.figure()
cp.plot_dist(src='storm')
plt.show()

xd, rd, = c.r_dist(2, 0.1, src='storm', norm_x=True)

x = c.data.data_dict['storm']['x']
y = c.data.data_dict['storm']['y']

plt.imshow(c.data.binary_img, interpolation='nearest')
plt.plot(x, y, linestyle='None', marker='.', color='r', alpha=0.2)
plt.show()

