from test.test_functions import generate_testdata
from colicoords import Cell, CellPlot
import matplotlib.pyplot as plt


data = generate_testdata('ds6')
c = Cell(data)
c.optimize()

x = c.data.data_dict['storm']['x']
y = c.data.data_dict['storm']['y']


print(y)

#
# new_data = data[10:50, 0:50]
# x = new_data.data_dict['storm']['x']
# y = new_data.data_dict['storm']['y']
# print(y)






cp = CellPlot(c)

fig, axes = plt.subplots(1, 2)
cp.plot_binary_img(ax=axes[0])
cp.plot_storm(ax=axes[0], data_name='storm', kernel='gaus', alpha=0.9)
axes[0].plot(x, y, linestyle='None', marker='.', color='r', alpha=0.9)
# cp.plot_outline()

r_data = c.data.rotate(50)
rc = Cell(r_data)


rcp = CellPlot(rc)

rx = rc.data.data_dict['storm']['x']
ry = rc.data.data_dict['storm']['y']

rcp.plot_binary_img(ax=axes[1])
rcp.plot_storm(ax=axes[1], data_name='storm', kernel='gaus', alpha=0.9)
axes[1].plot(rx, ry, linestyle='None', marker='.', color='r', alpha=0.9)
# cp.plot_outline()

plt.show()


