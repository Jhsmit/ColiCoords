from .cell import Cell,Coordinates
from .data_models import Data
import numpy as np
import mahotas as mh


def synth_cell(a0, a1, a2, xl, xr, r, pad_width=2):
    #todo choose a0 a1 a2 so that orientation is horizontal
    # shape = (a0*2 + 10, xr - xl + 2*r + 20)

    y_max = int(a0 + a1*xr + a2*xr**2)
    shape = (y_max + 10 + r, xr + 2*r + 10)

    coords = Coordinates(None, a0=a0, a1=a1, a2=a2, xl=xl, xr=xr, r=r, shape=shape, initialize=False)
    binary = coords.rc < r
    min1, max1, min2, max2 = mh.bbox(binary)
    min1p, max1p, min2p, max2p = min1 - pad_width, max1 + pad_width, min2 - pad_width, max2 + pad_width
    res = binary[min1p:max1p, 0:max2p]

    data = Data()
    data.add_data(res.astype(int), 'binary')
    cell = Cell(data)
    cell.coords.a0 = a0 - min1p
    cell.coords.a1 = a1
    cell.coords.a2 = a2
    cell.coords.xl = xl
    cell.coords.xr = xr
    cell.coords.r = r

    return cell


def add_img_modelled(cell, rmodel, dclass='fluorescence', name=None):
    x = np.linspace(0, np.max(cell.data.shape) / 1.8, num=25)
    y = rmodel(x)
    flu = np.interp(cell.coords.rc, x, y)
    cell.data.add_data(flu, dclass, name=name)

    return cell

