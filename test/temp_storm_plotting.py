from test.test_functions import load_stormdata
import matplotlib.pyplot as plt
from colicoords.preprocess import data_to_cells
from colicoords.plot import CellPlot
from colicoords.optimizers import STORMOptimizer
from colicoords import Data
from colicoords.fileIO import load_thunderstorm
from colicoords.config import cfg
import datetime
#tempshizzle
import numpy as np
import tifffile

from symfit import Fit, Parameter, Variable, exp

lacy_bin = tifffile.imread('temp_lacy/binary_out.tif')
storm_data = load_thunderstorm('temp_lacy/storm_table.csv', pixelsize=80/5)
#storm_data = np.genfromtxt('temp_lacy/storm_table.csv', skip_header=1, delimiter=',')

# data = Data()
# data.add_data(lacy_bin, 'binary')
# data.add_data(storm_data, 'storm')
#
# celllist_lacy = data_to_cells(data)
# print(len(celllist_lacy))
#
# lacy = celllist_lacy[0]
# lacy.optimize()
#
# so1 = STORMOptimizer(lacy, method='photons')
# so1.optimize_r()
# so1.optimize_endcaps()
# so1.optimize_fit()
# so1.optimize_overall()

data = load_stormdata()
cell_list = data_to_cells(data, rotate='binary')

c = cell_list[2]
c.optimize()
cp = CellPlot(c)

x = c.data.data_dict['storm']['x']
y = c.data.data_dict['storm']['y']

so = STORMOptimizer(c, method='photons')
so.optimize_r()
so.optimize_endcaps()
so.optimize_fit()
so.optimize_overall()

import seaborn as sns
sns.set_style('white')

# plt.plot(x, y, 'r.')
# plt.show()

x, neo_yvals = c.r_dist(30, 0.2, data_name='storm',)
print(len(x))
lacy = np.load('lacy_1_yvals.npy')

stop = 150
step = 1
bins = np.arange(0, stop+step, step)
xdata = (bins + 0.5 * 1)*(80./5.)

x = Variable(name='x')
y = Variable(name='y')
A = Parameter(name='A', value=1)#, value=200)
sig = Parameter(name='sig', value=50)#, value=50.0)
x0 = Parameter(name='x0', value=300)#, value=450.)

model = {y: A * exp(-(x - x0)**2/(2 * sig**2))}

fit = Fit(model, x=xdata[10:], y = (lacy / lacy.max())[10:])
res = fit.execute()

print(res)

lacy_fit = fit.model(np.linspace(0, xdata.max(), num=1000), **res.params).y

neo_x = (80/5)*np.arange(len(neo_yvals))

x = Variable(name='x')
y = Variable(name='y')
A = Parameter(name='A', value=1)#, value=200)
sig = Parameter(name='sig', value=50)#, value=50.0)
x0 = Parameter(name='x0', value=350)#, value=450.)

model = {y: A * exp(-(x - x0)**2/(2 * sig**2))}

fit = Fit(model, x=neo_x[10:], y = (neo_yvals / neo_yvals.max())[10:])
res = fit.execute()

print(res)
sns.set(font_scale=1.35)
sns.set_style('white')

neob_fit = fit.model(np.linspace(0, neo_x.max(), num=1000), **res.params).y


#plt.plot((80/5)*np.arange(len(neo_yvals)), neo_yvals, label='Neomycin C3 Rhodamine')


plt.plot((80/5)*np.arange(len(neo_yvals)), neo_yvals, label='Neomycin C3 Rhodamine')
#plt.plot(np.linspace(0, neo_x.max(), num=1000), neob_fit, color='k', linestyle='--', alpha=0.75)


#plt.plot(xdata, lacy / lacy.max(), label='Lacy eYFP')
#plt.plot(np.linspace(0, xdata.max(), num=1000), lacy_fit, color='k', linestyle='--', alpha=0.75)
plt.xlim(0, 1500)
plt.xlabel('Distance (nm)')
plt.ylabel('Number of localizations')
#plt.ylim(0, 1.1)
plt.legend()
plt.savefig('test1.png', dpi=300)
plt.show()

