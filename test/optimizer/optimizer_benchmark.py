import numpy as np
from colicoords.fileIO import load, save
from colicoords import Cell, CellList
from colicoords.optimizers import Optimizer
import time

import seaborn as sns
import matplotlib.pyplot as plt

#cell_list = load(r'../test_data/ds7/preoptimized_10.cc')
#c = cell_list[0]

# 'Newton-CG', 'dogleg', 'trust-ncg', 'trust-exact', 'trust-krylov' requires Jacobian
methods = ['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'L-BFGS-B', 'TNC',
          'COBYLA', 'SLSQP']


def worker(obj_list, **kwargs):
    for obj in obj_list:
        obj.optimize_mp(**kwargs)


if __name__ == '__main__':
    cl = load(r'../test_data/ds7/Neomycin.cc')

    cell_list = cl[:100].copy()

    before = np.mean([np.sum(np.logical_xor(c.data.binary_img, c.coords.rc < c.coords.r)) for c in cell_list])

    t0 = time.time()
    cell_list.optimize_mp()
    t1 = time.time()

    after = np.mean([np.sum(np.logical_xor(c.data.binary_img, c.coords.rc < c.coords.r)) for c in cell_list])

    print(before, after)
    print(t1 - t0)



    # s_result = []
    # s_time = []
    #
    # m_result = []
    # m_time = []
    # for n in n_cells:
    #     print(n)
    #     #serial
    #     cell_list = cl[:n].copy()
    #
    #     before = np.mean([np.sum(np.logical_xor(c.data.binary_img, c.coords.rc < c.coords.r)) for c in cell_list])
    #
    #     t0 = time.time()
    #     cell_list.optimize()
    #     t1 = time.time()
    #
    #     after = np.mean([np.sum(np.logical_xor(c.data.binary_img, c.coords.rc < c.coords.r)) for c in cell_list])
    #
    #     s_result.append((before, after))
    #     s_time.append(t1 - t0)
    #     print('serial', n, (before, after), t1 - t0)
    #
    #     time.sleep(10) # time for electron relaxation
    #
    #     #parallel
    #     cell_list = cl[:n].copy()
    #
    #     before = np.mean([np.sum(np.logical_xor(c.data.binary_img, c.coords.rc < c.coords.r)) for c in cell_list])
    #
    #     t0 = time.time()
    #     cell_list.optimize_mp()
    #     t1 = time.time()
    #
    #     after = np.mean([np.sum(np.logical_xor(c.data.binary_img, c.coords.rc < c.coords.r)) for c in cell_list])
    #
    #     m_result.append((before, after))
    #     m_time.append(t1 - t0)
    #     print('parallel', n, (before, after), t1 - t0)
    #
    #     time.sleep(10)
    #
    # final_arr = np.column_stack((n_cells, s_result, s_time, m_result, m_time))
    # print(final_arr)

    #np.savetxt('optimization_comparison.txt', final_arr)
    #np.save('optimization_comparison.npy', final_arr)

   #  final_arr = np.load('optimization_comparison.npy')
   #
   #  s_time = final_arr.T[3]
   #  m_time = final_arr.T[6]
   #
   #  sns.set(font_scale=2)
   #
   #  plt.figure()
   #  plt.plot(n_cells, s_time, label='serial')
   #  plt.plot(n_cells, m_time, label='parallel')
   #  plt.xlabel('Number of cells')
   #  plt.ylabel('Time (s)')
   #  plt.legend()
   #  plt.tight_layout()
   # # plt.show()
   #  plt.savefig('cells vs time.png')

    # 3.520260000228881836e+01

    #
    # processes = [3, 4, 5, 6]
    #
    # p_result = []
    # p_time = []
    # for p in processes:
    #     print(p)
    #     #serial
    #     cell_list = cl[:100].copy()
    #
    #     before = np.mean([np.sum(np.logical_xor(c.data.binary_img, c.coords.rc < c.coords.r)) for c in cell_list])
    #
    #     t0 = time.time()
    #     cell_list.optimize_mp(processes=p)
    #     t1 = time.time()
    #
    #     after = np.mean([np.sum(np.logical_xor(c.data.binary_img, c.coords.rc < c.coords.r)) for c in cell_list])
    #
    #     p_result.append((before, after))
    #     p_time.append(t1 - t0)
    #     print('processes', p, (before, after), t1 - t0)
    #
    #     time.sleep(10) # time for electron relaxation
    #
    # final_arr = np.column_stack((processes, p_result, p_time))
    # print(final_arr)

    # np.savetxt('optimization_comparison_proc.txt', final_arr)
    # np.save('optimization_comparison_proc.npy', final_arr)