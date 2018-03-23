import numpy as np
from scipy.ndimage.interpolation import rotate as scipy_rotate
import matplotlib.pyplot as plt
from colicoords.data_models import _rotate_storm

img_in = np.ones((400, 300))


angles = np.array([0, 10, 40, 45, 60, 80, 100, 120, 250, 270, 300, 350])


angles = np.append(angles, -angles)


st_in = np.array([(1,2) , (4,5), (6,7)], dtype=[('x', float), ('y', float)])
print(st_in['x'])


st_out = _rotate_storm(st_in, 0, shape=(10, 20))

print(st_out['x'])



# im_out = scipy_rotate(img_in, 40)
#
# plt.imshow(im_out, interpolation='none')
# plt.show()

#
#
# for th in angles:
#
#     th_rad = (np.pi/180)*th
#
#
#     im_out = scipy_rotate(img_in, -th)
#     print('th', th)
#     print(im_out.shape)
#
#
#     y, x = img_in.shape
#
#     a = np.abs(x * np.sin(th_rad)) + np.abs(y * np.cos(th_rad))
#     b = np.abs(x * np.cos(th_rad)) + np.abs(y * np.sin(th_rad))
#
#     y1, x1 = im_out.shape
#
#     print(a, b)
#     print(int(a), y1)
#     print(int(b), x1)
#     print(y1 == int(a))
#     print(x1 == int(b))
