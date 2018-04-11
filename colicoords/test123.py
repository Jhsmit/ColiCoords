import tables
import tqdm
import time
import numpy as np

# res_list = []
# for i in range(10):
#     d = {'r1': np.random.rand(), 'a1': np.random.rand()}
#     res_list.append(d)
#
#
#
# #list of dics to arrays
# arr = np.array(
#     [[d['r1'], d['a1']] for d in res_list]
# )
#
# print(arr.shape)
#
# np.savetxt('testfile.txt', arr, header = 'r1\ta1')


def _accept_test(bounds, **kwargs):
    par_values = kwargs['x_new']
    print(par_values)

    for (pmin, pmax), v in zip(bounds, par_values):
        print("----")
        print(v)
        print(pmin, pmax)

        print(pmin if pmin is not None else -np.inf)
        print(pmax if pmax is not None else np.inf)

        print(-np.inf if pmin is None else pmin)
        print(np.inf if pmax is None else pmax)

        print("----")

    bools = [(-np.inf if pmin is None else pmin) <= val <= (np.inf if pmax is None else pmax) for (pmin, pmax), val in
             zip(bounds, par_values)]
    print(bools)
    return np.all(bools)


bounds = [(None, 5), (0, 10)]
x = [2, -5]

_accept_test(bounds, x_new=x)