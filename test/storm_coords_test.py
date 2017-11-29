from test.test_functions import generate_testdata
from colicoords import Cell, CellPlot
import matplotlib.pyplot as plt


data = generate_testdata('ds6')
c = Cell(data)
c.optimize()

x = c.data.data_dict['storm']['x']
y = c.data.data_dict['storm']['y']

xt, yt = c.coords.transform(x, y, src='cart', tgt='mpl')

cp = CellPlot(c)
plt.figure()
cp.plot_binary_img()
cp.plot_storm('storm', alpha=0.9)
plt.plot(x, y, linestyle='None', marker='.', color='r', alpha=0.9)
# cp.plot_outline()


cp.show()