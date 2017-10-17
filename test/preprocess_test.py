from test.test_functions import generate_testdata
from colicoords.preprocess import batch_flu_images, cell_generator
import matplotlib.pyplot as plt
import numpy as np

data = generate_testdata()
print(np.unique(data.binary_img))
d = data[0]

print(d.binary_img.mean())
d1 = d.copy()
print('d1 fresh', d1.binary_img.mean())
data[0].binary_img += 20
print(d.binary_img.mean())
print(d1.binary_img.mean())

print(d1 == data[0])

