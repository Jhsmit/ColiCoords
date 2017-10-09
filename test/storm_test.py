from test.test_functions import generate_stormdata
import matplotlib.pyplot as plt

data = generate_stormdata()


plt.imshow(data.storm_img)
plt.show()



print(data.shape)