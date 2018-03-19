import numpy as np
from colicoords.fileIO import load, save
from colicoords import Cell, CellList
from colicoords.optimizers import Optimizer

#cell_list = load(r'../test_data/ds7/preoptimized_10.cc')
#c = cell_list[0]

# 'Newton-CG', 'dogleg', 'trust-ncg', 'trust-exact', 'trust-krylov' requires Jacobian
methods = ['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'L-BFGS-B', 'TNC',
          'COBYLA', 'SLSQP']


def worker(obj_list, **kwargs):
    for obj in obj_list:
        obj.optimize_mp(**kwargs)




if __name__ == '__main__':
    cl = load(r'../test_data/ds7/preoptimized_200.cc')

#    res = cl[0].optimize_mp()
#    print(res)
#    print(type(res[0]))

    #
    print(len(cl))
    CellList(cl[:50]).optimize_mp()



"""

if __name__ == '__main__':
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    cl = load(r'../test_data/ds7/preoptimized_200.cc')
    print('loaded')
    c = cl[0]
    print(len(cl))
    cl[0].optimize()
    before = np.mean([np.sum(np.logical_xor(c.data.binary_img, c.coords.rc < c.coords.r)) for c in cl])

    kwargs = {'data_name': 'binary', 'method': 'photons', 'verbose': False}
    iterable = list([(obj, kwargs) for obj in cl])

    #todo this works but doenst do shit
    #check the functionc can now be moved to a separate module avoiding name = main
    #rewrite optimization so that it returns stuff

    i1 = cl[:50]
    i2 = cl[50:100]
    i3 = cl[100:150]
    i4 = cl[150:]


    start_time = datetime.datetime.now()
    p1 = mp.Process(target=worker, args=(i1,), kwargs=kwargs)
    p2 = mp.Process(target=worker, args=(i2, ), kwargs=kwargs)
    p3 = mp.Process(target=worker, args=(i3, ), kwargs=kwargs)
    p4 = mp.Process(target=worker, args=(i4, ), kwargs=kwargs)

    p1.start()
    p2.start()
    p3.start()
    p4.start()

    p1.join()
    p2.join()
    p3.join()
    p4.join()


    # print(len(iterable))
    # pool = mp.Pool(processes=4)
    #
    # res = pool.imap(worker, iterable)
    # pool.close()
    # pool.join()
    # print(res)
    # print('hoidoei')

    time_diff = datetime.datetime.now() - start_time
    after = np.mean([np.sum(np.logical_xor(c.data.binary_img, c.coords.rc < c.coords.r)) for c in cl])
    print(before, after, time_diff)

    print('single process')
    cl = load(r'../test_data/ds7/preoptimized_200.cc')
    start_time = datetime.datetime.now()
    cl.optimize()
    time_diff = datetime.datetime.now() - start_time
    after = np.mean([np.sum(np.logical_xor(c.data.binary_img, c.coords.rc < c.coords.r)) for c in cl])

    print(before, after, time_diff)



    # cl.optimize()
    # after1 = np.mean([np.sum(np.logical_xor(c.data.binary_img, c.coords.rc < c.coords.r)) for c in cl])
    # print(before, after1)

    #
    # #start = datetime.datetime.now()
    # for method in ['Powell']: #methods[:1]:
    #     cell_list = CellList(load(r'../test_data/ds7/preoptimized_200.cc')[:25])
    #     before = np.mean([np.sum(np.logical_xor(c.data.binary_img, c.coords.rc < c.coords.r)) for c in cell_list])
    #
    #     print(method)
    #     start = datetime.datetime.now()
    #
    #     cell_list.optimize()
    #     #
    #     # for c in cell_list:
    #     #     bo = BinaryOptimizer(c)
    #     #     bo.optimize_overall(method=method, verbose=False)
    #     print('time:', datetime.datetime.now() - start)
    #
    #     after = np.mean([np.sum(np.logical_xor(c.data.binary_img, c.coords.rc < c.coords.r)) for c in cell_list])
    #     print(before, after)
    #     print('----')


#print('time:', datetime.datetime.now() - start)

"""