from .cell import Cell, Coordinates, _calc_len
from .data_models import Data
import numpy as np
import mahotas as mh
from scipy.integrate import quad
from scipy.optimize import fsolve


class SynthCell(Cell):
    def __init__(self, length, radius, curvature, pad_width=5, name=None):

        a2 = curvature
        xl = radius + pad_width
        xr = fsolve(calc_length, length, args=(xl, a2, length)).squeeze()
        a1 = -a2 * (xr + xl)
        r = radius
        xm = (xl + xr) / 2
        y_mid = a1*xm + a2*xm**2
        a0 = 4*radius - y_mid

        y_max = a0 + a1 * xr + a2 * xr ** 2
        shape = tuple(np.ceil([y_max + 10 + r, xr + 2 * r + 10]).astype(int))
        coords = Coordinates(None, a0=a0, a1=a1, a2=a2, xl=xl, xr=xr, r=r, shape=shape, initialize=False)
        binary = coords.rc < r
        min1, max1, min2, max2 = mh.bbox(binary)
        min1p, max1p, min2p, max2p = min1 - pad_width, max1 + pad_width, min2 - pad_width, max2 + pad_width
        res = binary[min1p:max1p, 0:max2p]

        data = Data()
        data.add_data(res.astype(int), 'binary')
        super(SynthCell, self).__init__(data, name=name)
        self.coords.a0 = a0 - min1p
        self.coords.a1 = a1
        self.coords.a2 = a2
        self.coords.xl = xl
        self.coords.xr = xr
        self.coords.r = r

    def add_radial_model_data(self, rmodel, dclass='fluorescence', name=None, **kwargs):
        #todo more catchy name for this function
        num = kwargs.pop('num', 25)
        x = np.linspace(0, np.max(self.data.shape) / 1.8, num=num)
        y = rmodel(x)
        flu = np.interp(self.coords.rc, x, y)
        self.data.add_data(flu, dclass, name=name)

    def add_storm_membrane(self, num, r_mean, r_std=None, name=None):
        def integrant_top(t, a1, a2, r):
            return np.sqrt(1 + (a1 + 2 * a2 * t) ** 2 + ((4 * a2 ** 2 * r ** 2) / (1 + (a1 + 2 * a2 * t) ** 2) ** 2) + (
                    (4 * a2 * r) / np.sqrt(1 + (a1 + 2 * a2 * t))))

        def integrant_bot(t, a1, a2, r):
            return np.sqrt(1 + (a1 + 2 * a2 * t) ** 2 + ((4 * a2 ** 2 * r ** 2) / (1 + (a1 + 2 * a2 * t) ** 2) ** 2) - (
                    (4 * a2 * r) / np.sqrt(1 + (a1 + 2 * a2 * t))))

        top, terr = quad(integrant_top, self.coords.xl, self.coords.xr, args=(self.coords.a1, self.coords.a2, r_mean))
        bot, berr = quad(integrant_bot, self.coords.xl, self.coords.xr, args=(self.coords.a1, self.coords.a2, r_mean))

        segments_lenghts = np.array([np.pi * r_mean, top, np.pi * r_mean, bot])
        total = np.sum(segments_lenghts)
        cumsum = np.cumsum(segments_lenghts)

        s = np.random.uniform(0, np.nextafter(total, total + 1), num)
        i = np.digitize(s, cumsum)

        s_rel = s - np.insert(cumsum, 0, 0)[i]

        x_res = np.empty_like(s_rel)
        y_res = np.empty_like(s_rel)
        new_r = np.random.normal(loc=r_mean, scale=r_std, size=num) if r_std else r_mean*np.ones(num)

        #i == 0
        th1 = np.arctan(self.coords.p_dx(self.coords.xl))
        th2 = s_rel[i == 0] / r_mean
        x_res[i == 0] = self.coords.xl + new_r[i == 0]*np.sin(-th2 - th1)
        y_res[i == 0] = self.coords.p(self.coords.xl) + new_r[i == 0]*np.cos(-th2 - th1)

        #i == 1
        t = (s_rel[i == 1] / top) * (self.coords.xr - self.coords.xl) + self.coords.xl
        x_res[i == 1] = t + new_r[i == 1] * ((self.coords.a1 + 2 * self.coords.a2 * t) /
                                             np.sqrt(1 + (self.coords.a1 + 2 * self.coords.a2 * t) ** 2))
        y_res[i == 1] = self.coords.a0 + self.coords.a1 * t + self.coords.a2 * (t ** 2) - new_r[i == 1] * (1 / np.sqrt(1 + (self.coords.a1 + 2 * self.coords.a2 * t) ** 2))

        #i == 2
        th1 = np.arctan(self.coords.p_dx(self.coords.xr))
        th2 = s_rel[i == 2] / r_mean

        x_res[i == 2] = self.coords.xr + new_r[i == 2]*np.sin(th1 + th2)
        y_res[i == 2] = self.coords.p(self.coords.xr) - new_r[i == 2]*np.cos(th1 + th2)

        #i == 3
        t = self.coords.xr - (s_rel[i == 3] / bot) * (self.coords.xr - self.coords.xl)
        x_res[i == 3] = t + - new_r[i == 3] * ((self.coords.a1 + 2 * self.coords.a2 * t) / np.sqrt(1 + (self.coords.a1 + 2 * self.coords.a2 * t) ** 2))
        y_res[i == 3] = self.coords.a0 + self.coords.a1 * t + self.coords.a2 * (t ** 2) + new_r[i == 3] * (1 / np.sqrt(1 + (self.coords.a1 + 2 * self.coords.a2 * t) ** 2))

        storm = np.recarray((len(x_res, )), dtype=[('x', float), ('y', float), ('frame', int), ('intensity', int)])
        storm['x'] = x_res
        storm['y'] = y_res
        storm['frame'] = np.zeros_like(x_res)
        storm['intensity'] = np.ones_like(x_res)
        self.data.add_data(storm, 'storm', name=name)

        return storm
    # #i == 0
    # th1 = np.arctan(cell.coords.p_dx(xl))
    # th2 = s_rel[i == 0] / r
    # x_res[i == 0] = xl + r*np.sin(-th2 - th1)
    # y_res[i == 0] = cell.coords.p(xl) + r*np.cos(-th2 - th1)
    #
    # #i == 1
    # t = (s_rel[i == 1] / top) * (xr - xl) + xl
    # x_res[i == 1] = t + r * ((a1 + 2 * a2 * t) / np.sqrt(1 + (a1 + 2 * a2 * t) ** 2))
    # y_res[i == 1] = a0 + a1 * t + a2 * (t ** 2) - r * (1 / np.sqrt(1 + (a1 + 2 * a2 * t) ** 2))
    #
    # #i == 2
    # th1 = np.arctan(cell.coords.p_dx(xr))
    # th2 = s_rel[i == 2] / r
    # # x =
    # # y =
    # x_res[i == 2] = xr + r*np.sin(th1 + th2)
    # y_res[i == 2] = cell.coords.p(xr) - r*np.cos(th1 + th2)
    #
    # #i == 3
    # t = xr - (s_rel[i == 3] / bot) * (xr - xl)
    #
    # x_res[i == 3] = t + - r * ((a1 + 2 * a2 * t) / np.sqrt(1 + (a1 + 2 * a2 * t) ** 2))
    # y_res[i == 3] = a0 + a1 * t + a2 * (t ** 2) + r * (1 / np.sqrt(1 + (a1 + 2 * a2 * t) ** 2))


def calc_length(xr, xl, a2, length):
    a1 = -a2 * (xr + xl)
    l = (1 / (4 * a2)) * (
            ((a1 + 2 * a2 * xr) * np.sqrt(1 + (a1 + 2 * a2 * xr) ** 2) + np.arcsinh((a1 + 2 * a2 * xr))) -
            ((a1 + 2 * a2 * xl) * np.sqrt(1 + (a1 + 2 * a2 * xl) ** 2) + np.arcsinh((a1 + 2 * a2 * xl)))
    )

    return length-l


def synth_cell(a0, a1, a2, xl, xr, r, pad_width=2):
    #todo choose a0 a1 a2 so that orientation is horizontal
    # shape = (a0*2 + 10, xr - xl + 2*r + 20)

    y_max = a0 + a1*xr + a2*xr**2
    print(y_max, r, xr)
    shape = tuple(np.ceil([y_max + 10 + r, xr + 2*r + 10]).astype(int))
    print('shape', shape)
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

#todo needs some refactoring into frigging class!!!!!!!!!oneeleven
def get_storm_membrane(cell, a0, a1, a2, xl, xr, r, num):
    def integrant_top(t, a1, a2, r):
        return np.sqrt(1 + (a1 + 2 * a2 * t) ** 2 + ((4 * a2 ** 2 * r ** 2) / (1 + (a1 + 2 * a2 * t) ** 2) ** 2) + (
                (4 * a2 * r) / np.sqrt(1 + (a1 + 2 * a2 * t))))

    def integrant_bot(t, a1, a2, r):
        return np.sqrt(1 + (a1 + 2 * a2 * t) ** 2 + ((4 * a2 ** 2 * r ** 2) / (1 + (a1 + 2 * a2 * t) ** 2) ** 2) - (
                (4 * a2 * r) / np.sqrt(1 + (a1 + 2 * a2 * t))))

    top, terr = quad(integrant_top, xl, xr, args=(a1, a2, r))
    bot, berr = quad(integrant_bot, xl, xr, args=(a1, a2, r))

    segments_lenghts = np.array([np.pi * r, top, np.pi * r, bot])
    total = np.sum(segments_lenghts)
    cumsum = np.cumsum(segments_lenghts)

    s = np.random.uniform(0, np.nextafter(total, total + 1), num)
    i = np.digitize(s, cumsum)

    s_rel = s - np.insert(cumsum, 0, 0)[i]

    x_res = np.empty_like(s_rel)
    y_res = np.empty_like(s_rel)

    new_r = np.random.normal(loc=r, scale=0.25, size=num)

    #i == 0
    th1 = np.arctan(cell.coords.p_dx(xl))
    th2 = s_rel[i == 0] / r
    x_res[i == 0] = xl + new_r[i == 0]*np.sin(-th2 - th1)
    y_res[i == 0] = cell.coords.p(xl) + new_r[i == 0]*np.cos(-th2 - th1)

    #i == 1
    t = (s_rel[i == 1] / top) * (xr - xl) + xl
    x_res[i == 1] = t + new_r[i == 1] * ((a1 + 2 * a2 * t) / np.sqrt(1 + (a1 + 2 * a2 * t) ** 2))
    y_res[i == 1] = a0 + a1 * t + a2 * (t ** 2) - new_r[i == 1] * (1 / np.sqrt(1 + (a1 + 2 * a2 * t) ** 2))

    #i == 2
    th1 = np.arctan(cell.coords.p_dx(xr))
    th2 = s_rel[i == 2] / r

    x_res[i == 2] = xr + new_r[i == 2]*np.sin(th1 + th2)
    y_res[i == 2] = cell.coords.p(xr) - new_r[i == 2]*np.cos(th1 + th2)

    #i == 3
    t = xr - (s_rel[i == 3] / bot) * (xr - xl)
    x_res[i == 3] = t + - new_r[i == 3] * ((a1 + 2 * a2 * t) / np.sqrt(1 + (a1 + 2 * a2 * t) ** 2))
    y_res[i == 3] = a0 + a1 * t + a2 * (t ** 2) + new_r[i == 3] * (1 / np.sqrt(1 + (a1 + 2 * a2 * t) ** 2))

    return x_res, y_res

class SynthCellList(object):
    def __init__(self, num, radii, lenghts, curvatures):
