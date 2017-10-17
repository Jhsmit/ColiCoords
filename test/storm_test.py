from test.test_functions import generate_stormdata
import matplotlib.pyplot as plt
from cellcoordinates.preprocess import data_to_cells
data = generate_stormdata()
import numpy as np
print(len(data))
for d in data:
    print('woei')

# plt.imshow(data.storm_img)
# plt.show()

x = data.storm_table['x'] / 80
y = data.storm_table['y'] / 80
#todo type
print(len(data.storm_table))

xt, yt = data.transform(x, y, src='mpl', tgt='cart')

#
# plt.imshow(data.binary_img)
# plt.plot(xt, yt, linestyle='None', marker='.', color='r', alpha=0.2)
# #
# plt.show()


d = data[50:100, 100:150]

print('ndim', data.data_dict['Binary'].ndim)

cell_list = data_to_cells(data, rotate='Binary')
print(len(cell_list))
print(data.shape)

c = cell_list[2]
x = c.data.storm_table['x']
y = c.data.storm_table['y']


stop = 20
step = 0.25
bins = np.arange(0, stop+step, step)
r = c.coords.calc_rc(x, y)
i_sort = r.argsort()
r_sorted = r[i_sort]
print(r_sorted)
bininds = np.digitize(r_sorted, bins)
yvals = np.bincount(bininds, minlength=len(bins))

print('hoiii')



plt.imshow(c.data.binary_img)
plt.plot(x, y, linestyle='None', marker='.', color='r', alpha=0.2)
plt.show()



plt.figure()
plt.plot(yvals)
plt.show()



x, y = c.radial_distribution(2, 0.1, src='STORMTable', storm_weight='points', norm_x=True)

plt.figure()
plt.plot(x, y)
plt.show()