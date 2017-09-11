from test.test_functions import generate_testdata
from cellcoordinates.preprocess import batch_flu_images, cell_generator
import matplotlib.pyplot as plt
import numpy as np

data = generate_testdata()
print(np.unique(data.binary_img))
d = data[0]


cells = [c for c in cell_generator(d.binary_img, d.flu_dict, rotate='binary')]

plt.imshow(cells[0].data.binary_img)
plt.show()