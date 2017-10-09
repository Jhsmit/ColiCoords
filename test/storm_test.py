from test.test_functions import generate_stormdata
import matplotlib.pyplot as plt
from cellcoordinates.preprocess import data_to_cells
data = generate_stormdata()
print(len(data))
for d in data:
    print('woei')

# plt.imshow(data.storm_img)
# plt.show()

x = data.storm_table['x'] / 80
y = data.storm_table['y'] / 80

print(len(data.storm_table))

xt, yt = data.transform(x, y, src='mpl', tgt='cart')

#
# plt.imshow(data.binary_img)
# plt.plot(xt, yt, linestyle='None', marker='.', color='r', alpha=0.2)
# #
# plt.show()


d = data[50:100, 100:150]



cell_list = data_to_cells(data, rotate='Binary')
print(len(cell_list))
print(data.shape)

c = cell_list[3]
x = c.data.storm_table['x'] / 80
y = c.data.storm_table['y'] / 80


plt.imshow(c.data.binary_img)
plt.plot(x, y, linestyle='None', marker='.', color='r', alpha=0.2)

plt.show()