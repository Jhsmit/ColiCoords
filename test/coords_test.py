import numpy as np
import matplotlib.pyplot as plt


img = np.arange(32).reshape(4, -1)

plt.imshow(img, interpolation='none', extent=[0, 8, 4, 0])
plt.show()