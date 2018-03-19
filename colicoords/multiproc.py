#import multiprocessing as mp
import multiprocess as mp
import datetime
import numpy as np
from functools import partial




def worker(obj_list, **kwargs):
    for obj in obj_list:
        obj.optimize_mp(**kwargs)

def optimimize_multiprocess(cell_list, data_name='binary', method='photons', verbose=False):
    before = np.mean([np.sum(np.logical_xor(c.data.binary_img, c.coords.rc < c.coords.r)) for c in cell_list])

    kwargs = {'data_name': data_name, 'method': method, 'verbose': verbose}

    from colicoords.cell import CellList

    i1 = CellList(cell_list[:50])
    i2 = CellList(cell_list[50:100])
    i3 = CellList(cell_list[100:150])
    i4 = CellList(cell_list[150:])

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

    time_diff = datetime.datetime.now() - start_time
    after = np.mean([np.sum(np.logical_xor(c.data.binary_img, c.coords.rc < c.coords.r)) for c in cell_list])
    print(before, after, time_diff)


def worker_single(obj, **kwargs):
    return obj.optimize(**kwargs)

def optimimize_multiprocess_mk2(cell_list, data_name='binary', method='photons', verbose=False):
    before = np.mean([np.sum(np.logical_xor(c.data.binary_img, c.coords.rc < c.coords.r)) for c in cell_list])

    kwargs = {'data_name': data_name, 'method': method, 'verbose': verbose}

    pool = mp.Pool()
    iterable = list([(obj, kwargs) for obj in cell_list])

    res = pool.starmap_async(worker_single, iterable)
    pool.close()
    pool.join()

    print('hoidoei')


def optimimize_multiprocess_mk3(cell_list, data_name='binary', objective=None, **kwargs):
    before = np.mean([np.sum(np.logical_xor(c.data.binary_img, c.coords.rc < c.coords.r)) for c in cell_list])

    kwargs = {'data_name': data_name, 'objective': objective, **kwargs}

    pool = mp.Pool(8)
    iterable = list([(obj, kwargs) for obj in cell_list])

    print(iterable[1])

    f = partial(worker_single, **kwargs)

    #res = pool.starmap_async(worker_single, iterable)
    res = pool.map(f, cell_list)


    for (r,v), cell in zip(res, cell_list):
        print(r, cell.name)
        cell.coords.sub_par(r)


    after = np.mean([np.sum(np.logical_xor(c.data.binary_img, c.coords.rc < c.coords.r)) for c in cell_list])

    print(before, after)

    # pool.close()
    # pool.join()



    print('hoidoei')