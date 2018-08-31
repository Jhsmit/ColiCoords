from .cell import Cell, Coordinates, CellList
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
        if num <= 0:
            x_res = np.array([])
            y_res = np.array([])
        else:
            # move theese definitions
            def integrant_top(t, a1, a2, r):
                return np.sqrt(1 + (a1 + 2 * a2 * t) ** 2 + ((4 * a2 ** 2 * r ** 2) / (1 + (a1 + 2 * a2 * t) ** 2) ** 2) + (
                        (4 * a2 * r) / np.sqrt(1 + (a1 + 2 * a2 * t))))

            def integrant_bot(t, a1, a2, r):
                return np.sqrt(1 + (a1 + 2 * a2 * t) ** 2 + ((4 * a2 ** 2 * r ** 2) / (1 + (a1 + 2 * a2 * t) ** 2) ** 2) - (
                        (4 * a2 * r) / np.sqrt(1 + (a1 + 2 * a2 * t))))

            top, terr = quad(integrant_top, self.coords.xl, self.coords.xr, args=(self.coords.a1, self.coords.a2, r_mean))
            bot, berr = quad(integrant_bot, self.coords.xl, self.coords.xr, args=(self.coords.a1, self.coords.a2, r_mean))

            segments_lenghts = np.array([np.pi * r_mean, top, np.pi * r_mean, bot])
            if np.any(np.isnan(segments_lenghts)):
                print(' vallerrlrlr' )
                raise ValueError('uhoh')
            print(segments_lenghts)
            print(np.any(segments_lenghts == np.nan))
            print(segments_lenghts[1] == np.nan)
            print(type(segments_lenghts[1]))
            print(segments_lenghts[1])

            total = np.sum(segments_lenghts)
            cumsum = np.cumsum(segments_lenghts)

            print('total', total, num)
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

    def gen_storm_image(self, intensities, sigma, data_elem='storm'):
        storm_table = self.data.data_dict[data_elem]
        img = np.zeros(self.coords.shape)
        for _int, storm_row in zip(intensities, storm_table):
            storm_row['intensity'] = _int
            mu_x = storm_row['x']
            mu_y = storm_row['y']

            img += _int*np.exp(-(((mu_x-self.coords.x_coords)/sigma)**2+((mu_y-self.coords.y_coords)/sigma)**2)/2)

        return img


def calc_length(xr, xl, a2, length):
    a1 = -a2 * (xr + xl)
    l = (1 / (4 * a2)) * (
            ((a1 + 2 * a2 * xr) * np.sqrt(1 + (a1 + 2 * a2 * xr) ** 2) + np.arcsinh((a1 + 2 * a2 * xr))) -
            ((a1 + 2 * a2 * xl) * np.sqrt(1 + (a1 + 2 * a2 * xl) ** 2) + np.arcsinh((a1 + 2 * a2 * xl)))
    )

    return length-l


class SynthCellList(CellList):
    def __init__(self, radii, lengths, curvatures):
        cell_list = [SynthCell(l, r, c, name='Cell_' + str(i).zfill(int(np.ceil(np.log10(len(radii)))))) for i, (l, r, c) in enumerate(zip(radii, lengths, curvatures))]
        super(SynthCellList, self).__init__(cell_list)