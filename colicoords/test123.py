import tables
import tqdm
import time
import numpy as np

res_list = []
for i in range(10):
    d = {'r1': np.random.rand(), 'a1': np.random.rand()}
    res_list.append(d)



#list of dics to arrays
arr = np.array(
    [[d['r1'], d['a1']] for d in res_list]
)

print(arr.shape)

np.savetxt('testfile.txt', arr, header = 'r1\ta1')