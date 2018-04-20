#https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.basinhopping.html#scipy.optimize.basinhopping

import mahotas as mh
import numpy as np
import operator
from functools import partial
from colicoords.optimizers import Optimizer
#import multiprocessing as mp
import multiprocess as mp
from tqdm import tqdm
import sys


# todo obj or class? in docstring
class Cell(object):
    """ ColiCoords' main single-cell object


    Attributes:
        data (:class:`Data`): Holds all data describing this single cell.
        coords (:class:`Coordinates`): Calculates and optimizes the cell's coordinate system.
    """

    def __init__(self, data_object, name=None):
        """
        Args:
            data_object (:class:`Data`): Data class holding all data which describes this single cell
            name (:obj:`str`): Name to identify this single cell.
        """

        self.data = data_object
        self.coords = Coordinates(self.data)
        self.name = name

    def optimize(self, data_name='binary', objective=None, **kwargs):
        """ Docstring will be added when all optimization types are supported

        Args:
            data_name (:obj:`str`):
            method:
            verbose:
        """
        optimizer = Optimizer(self, data_name=data_name, objective=objective)
        return optimizer.optimize(**kwargs)

    @property
    def radius(self):
        """float: Radius of the cell"""
        return self.coords.r

    @property
    def length(self):
        """float: Length of the cell obtained by integration of the spine arc length from xl to xr"""
        a0, a1, a2 = self.coords.coeff
        xl, xr = self.coords.xl, self.coords.xr
        l = (1 / (4 * a2)) * (
            ((a1 + 2 * a2 * xr) * np.sqrt(1 + (a1 + 2 * a2 * xr) ** 2) + np.arcsinh((a1 + 2 * a2 * xr))) -
            ((a1 + 2 * a2 * xl) * np.sqrt(1 + (a1 + 2 * a2 * xl) ** 2) + np.arcsinh((a1 + 2 * a2 * xl)))
        )

        return l

    @property
    def area(self):
        """float: Area of the cell in squared pixels"""
        return 2*self.length*self.coords.r + np.pi*self.coords.r**2

    @property
    def volume(self):
        """float: Volume of the cell in cubic pixels"""
        return np.pi*self.coords.r**2*self.length + (4/3)*np.pi*self.coords.r**3

    def a_list(self):
        raise NotImplementedError()

    def l_dist(self, norm_x=False):
        pass

    def r_dist(self, stop, step, data_name='', norm_x=False, storm_weight='points', xlim=None):
        #todo test xlim!
        """ Calculates the radial distribution of a given data element

        Args:
            stop: Until how far from the cell spine the radial distribution should be calculated
            step: The binsize of the returned radial distribution
            data_name: The name of the data element on which to calculate the radial distribution
            norm_x: If `True` the returned distribution will be normalized with the cell's radius set to 1.
            storm_weight: When calculating the radial distribution of  # todo change to True/False

        Returns:
            (tuple): tuple containing:

                xvals (:class:`np.ndarray`) Array of distances from the cell midline, values are the middle of the bins
                yvals (:class:`np.ndarray`) Array of in heights
        """

        # def bin_func(r, y_weight, bins):
        #     i_sort = r.argsort()
        #     r_sorted = r[i_sort]
        #     y_weight = y_weight[i_sort] if y_weight is not None else y_weight
        #     bin_inds = np.digitize(r_sorted,
        #                            bins) - 1  # -1 to assure points between 0 and step are in bin 0 (the first)
        #     yvals = np.bincount(bin_inds, weights=y_weight, minlength=len(bins))
        #     if y_weight is not None:
        #         yvals /= np.bincount(bin_inds, minlength=len(bins))
        #     return np.nan_to_num(yvals)

        bins = np.arange(0, stop+step, step)
        xvals = bins + 0.5 * step  # xval is the middle of the bin
        if not data_name:
            data_elem = list(self.data.flu_dict.values())[0] #yuck
        else:
            try:
                data_elem = self.data.data_dict[data_name]
            except KeyError:
                raise ValueError('Chosen data not found')

        if data_elem.ndim == 1:
            assert data_elem.dclass == 'storm'
            x = data_elem['x']
            y = data_elem['y']

            r = self.coords.calc_rc(x, y)
            r = r / self.coords.r if norm_x else r

            if xlim:
                xc = self.coords.calc_xc(x, y)
                if xlim == 'full':
                    b = (xc > self.coords.xl) * (xc < self.coords.xr).astype(bool)
                else:
                    mid_x = (self.coords.xl + self.coords.xr) / 2
                    b = (xc > mid_x - xlim) * (xc < mid_x + xlim).astype(bool)

                bin_r = r[b].flatten()
            else:
                bin_r = r.flatten()
                b = np.ones_like(x).astype(bool)


            if storm_weight == 'points':
                y_weight = None
            elif storm_weight == 'photons':
                y_weight = data_elem['intensity'][b]
            else:
                raise ValueError("storm_weights has to be either 'points' or 'photons'")
            yvals = self._bin_func(bin_r, y_weight, bins)

        elif data_elem.ndim == 2:
            #assert data_elem.dclass == 'fluorescence'

            r = self.coords.rc / self.coords.r if norm_x else self.coords.rc
            if xlim:
                if xlim == 'full':
                    b = (self.coords.xc > self.coords.xl) * (self.coords.xc < self.coords.xr).astype(bool)
                else:
                    mid_x = (self.coords.xl + self.coords.xr)/2
                    b = (self.coords.xc > mid_x - xlim)*(self.coords.xc < mid_x + xlim).astype(bool)

                bin_r = r[b].flatten()
                y_weight = data_elem[b].flatten()
            else:
                bin_r = r.flatten()
                y_weight = data_elem.flatten()


            yvals = self._bin_func(bin_r, y_weight, bins)
        elif data_elem.ndim == 3:  # todo check if this still works
            r = self.coords.rc / self.coords.r if norm_x else self.coords.rc
            yvals = np.vstack([self._bin_func(r.flatten(), d.flatten(), bins) for d in data_elem])
        else:
            raise ValueError('Invalid fluorescence image dimensions')
        return xvals, yvals

    def sim_cell(self, data_name, norm_x=False, r_scale=1, **kwargs):
        data_elem = self.data.data_dict[data_name]
        stop = kwargs.pop('stop', 25)  # todo default in cfg?
        step = kwargs.pop('step', 1) # todo cfg

        xp, fp = self.r_dist(stop, step, data_name=data_name, norm_x=norm_x) #todo what about norm_x?


        interp = np.interp(r_scale*self.coords.rc, xp, np.nan_to_num(fp))

        return interp


    @staticmethod
    def _bin_func(xvals, y_weight, bins):
        i_sort = xvals.argsort()
        r_sorted = xvals[i_sort]
        y_weight = y_weight[i_sort] if y_weight is not None else y_weight
        bin_inds = np.digitize(r_sorted,
                               bins) - 1  # -1 to assure points between 0 and step are in bin 0 (the first)
        yvals = np.bincount(bin_inds, weights=y_weight, minlength=len(bins))
        if y_weight is not None:
            yvals /= np.bincount(bin_inds, minlength=len(bins))
        return np.nan_to_num(yvals)

    def get_intensity(self, mask='binary', data_name='') -> float:
        """ Returns the mean fluorescence intensity in the region masked by either the binary image or synthetic
            binary image derived from the cell's coordinate system

        Args:
            mask (:obj:`str`): Either 'binary' or 'coords' to specify the source of the mask used
            data_name (:obj:`str`): The name of the fluorescence image data element to get the intensity values from.

        Returns:
            Mean fluorescence pixel value

        """
        if mask == 'binary':
            m = self.data.binary_img.astype(bool)
        elif mask == 'coords':
            m = self.coords.rc < self.coords.r
        else:
            raise ValueError("mask keyword should be either 'binary' or 'coords'")

        if not data_name:
            data_elem = list(self.data.flu_dict.values())[0] #yuck
        else:
            try:
                data_elem = self.data.data_dict[data_name]
            except KeyError:
                raise ValueError('Chosen data not found')

        return data_elem[m].mean()

    def copy(self):
        #todo needs testing
        new_cell = Cell(data_object=self.data.copy(), name=self.name)
        for par in self.coords.parameters:
            setattr(new_cell.coords, par, getattr(self.coords, par))

        return new_cell


class Coordinates(object):
    """Cell's coordinate system described by the polynomial p(x) and associated functions



    Attributes:
        xl (float): Left cell pole x-coordinate
        xr (float): Right cell pole x-coordinate
        r (float): Cell radius
        coeff (:class: ~numpy.ndarray): Coefficients [a0, a1, a2] of the polynomial a0 + a1*x + a2*x**2 which describes
        the cell's shape

    """

    parameters = ['r', 'xl', 'xr', 'a0', 'a1', 'a2']

    def __init__(self, data):
        """

        Args:
            data:
        """
        self.data = data
        self.coeff = np.array([1., 1., 1.])
        self.xl, self.xr, self.r, self.coeff = self._initial_guesses(data) #refactor to class method
        self.coeff = self._initial_fit()
        self.shape = data.shape

    #todo maybe remove this and instead store parameters as individual attributes:
    @property
    def a0(self):
        return self.coeff[0]

    @a0.setter
    def a0(self, value):
        self.coeff[0] = value

    @property
    def a1(self):
        return self.coeff[1]

    @a1.setter
    def a1(self, value):
        self.coeff[1] = value

    @property
    def a2(self):
        return self.coeff[2]

    @a2.setter
    def a2(self, value):
        self.coeff[2] = value

    def sub_par(self, par_dict):
        for k, v in par_dict.items():
            setattr(self, k, v)

    def calc_xc(self, xp, yp):
        """ Calculates the coordinate xc on p(x) closest to xp, yp
        
        All coordinates are cartesian. Solutions are found by solving the cubic equation.

        Args:
            xp: Input scalar or vector/matrix x-coordinate. Must be the same shape as yp 
            yp: Input scalar or vector/matrix x-coordinate. Must be the same shape as xp

        Returns:
            Scalar or vector/matrix depending on input
        """

        #https://en.wikipedia.org/wiki/Cubic_function#Algebraic_solution
        a0, a1, a2 = self.coeff
        #xp, yp = xp.astype('float32'), yp.astype('float32')
        # Converting of cell spine polynomial coefficients to coefficients of polynomial giving distance r
        a, b, c, d = 4*a2**2, 6*a1*a2, 4*a0*a2 + 2*a1**2 - 4*a2*yp + 2, 2*a0*a1 - 2*a1*yp - 2*xp
        #a: float, b: float, c: array, d: array
        discr = 18*a*b*c*d - 4*b**3*d + b**2*c**2 - 4*a*c**3 - 27*a**2*d**2

        #@jit(cache=True, nopython=True)
        #@profile
        def solve_general_bak(a, b, c, d):
            """
            Solve cubic polynomial in the form a*x^3 + b*x^2 + c*x + d
            Only works if polynomial discriminant < 0, then there is only one real root which is the one that is returned.
            https://en.wikipedia.org/wiki/Cubic_function#General_formula
            :return (float): Only real root
            """

            d0 = (b ** 2 - 3 * a * c)
            d1 = (2 * b ** 3 - 9 * a * b * c + 27 * a ** 2 * d)

            dc = ((d1 + (d1 ** 2 - 4 * d0 ** 3)**(1/2)) / 2)**(1/3)

            return -(1 / (3 * a)) * (b + dc + (d0 / dc))


        #@profile
        def solve_general(a, b, c, d):
            """
            Solve cubic polynomial in the form a*x^3 + b*x^2 + c*x + d
            Only works if polynomial discriminant < 0, then there is only one real root which is the one that is returned.
            https://en.wikipedia.org/wiki/Cubic_function#General_formula
            :return (float): Only real root
            """

            #todo check type for performance gain?
            # 16 16: 5.03 s
            # 32 32: 3.969 s
            # 64 64: 5.804 s
            # 8 8:
            d0 = (b ** 2. - 3. * a * c)#.astype('float32')
            d1 = (2. * b ** 3. - 9. * a * b * c + 27. * a ** 2. * d)#.astype('float32')

            #r0 = (d1 ** 2. - 4. * d0 ** 3.)
            r0 = (np.square(d1) - 4. * d0 ** 3.)
            #r1 = (d1 + np.sqrt(d1 ** 2. - 4. * d0 ** 3.)) / 2
            r1 = (d1 + np.sqrt(r0)) / 2
            #r1 = (d1 + np.sqrt(np.square(d1, 2.) - 4. * np.power(d0, 3.))) / 2
            dc = np.cbrt(r1)
            #dc = r1**(1/3)# to power (1/3) gives nan's for coeffs [1.98537881e+01, 1.44894594e-02, 2.38096700e+00]
            #dc = np.power(r1, 1/3)
            #dc = np.cbrt((d1 + np.sqrt(d1 ** 2. - 4. * d0 ** 3.)) / 2.)
            return -(1. / (3. * a)) * (b + dc + (d0 / dc))
            #todo hit a runtimewaring divide by zero on line above once

        def solve_trig(a, b, c, d):
            """
            Solve cubic polynomial in the form a*x^3 + b*x^2 + c*x + d
            https://en.wikipedia.org/wiki/Cubic_function#Trigonometric_solution_for_three_real_roots
            Only works if polynomial discriminant > 0, the polynomial has three real roots
            :return (float): 1st real root
            """

            p = (3. * a * c - b ** 2.) / (3. * a ** 2.)
            q = (2. * b ** 3. - 9. * a * b * c + 27. * a ** 2. * d) / (27. * a ** 3.)
            assert(np.all(p < 0))
            k = 0.
            t_k = 2. * np.sqrt(-p/3.) * np.cos((1 / 3.) * np.arccos(((3.*q)/(2.*p)) * np.sqrt(-3./p)) - (2*np.pi*k)/3.)
            x_r = t_k - (b/(3*a))
            assert(np.all(x_r > 0)) # dont know if this is guaranteed otherwise boundaries need to be passed and choosing from 3 slns
            return x_r

        if np.any(discr == 0):
            raise ValueError('Discriminant equal to zero encountered. This has never happened before! What did you do?')

        if np.all(discr < 0):
            x_c = solve_general(a, b, c, d)
        else:
            x_c = np.zeros(xp.shape)
            mask = discr < 0

            general_part = solve_general(a, b, c[mask], d[mask])
            trig_part = solve_trig(a, b, c[~mask], d[~mask])

            x_c[mask] = general_part
            x_c[~mask] = trig_part


       # print(solve_general.inspect_types())
        return x_c

    def calc_rc(self, xp, yp):
        """
        Applies endcap limits xl and xr and calculates distance r to cell spine
        :param xp: 1D array of x coordinates
        :param yp: 1D array of y coordinates
        :return: 1D array of distances r from (x, y) to (xc, p(xc))
        """
        xc = self.calc_xc(xp, yp)

        # # this is not strictly correct! also requires y coordinates
        # xc[xc < self.xl] = self.xl
        # xc[xc > self.xr] = self.xr

        # Area left of perpendicular line at xl:
        op = operator.lt if self.p_dx(self.xl) > 0 else operator.gt
        xc[op(yp, self.q(xc, self.xl))] = self.xl

        # Area right of perpendicular line at xr:
        op = operator.gt if self.p_dx(self.xr) > 0 else operator.lt
        xc[op(yp, self.q(xc, self.xr))] = self.xr

        a0, a1, a2 = self.coeff
        return np.sqrt((xc - xp)**2 + (a0 + xc*(a1 + a2*xc) - yp)**2)

    def calc_psi(self, xp, yp):
        return

    #todo check this 05 buissisnesese
    @property
    def x_coords(self):
        """ obj:`np.ndarray`: Matrix of shape m x n equal to cell image with cartesian x-coordinates."""
        ymax = self.shape[0]
        xmax = self.shape[1]
        return np.repeat(np.arange(xmax), ymax).reshape(xmax, ymax).T + 0.5

    @property
    def y_coords(self):
        """ obj:`np.ndarray`: Matrix of shape m x n equal to cell image with cartesian y-coordinates."""
        ymax = self.shape[0]
        xmax = self.shape[1]
        return np.repeat(np.arange(ymax), xmax).reshape(ymax, xmax) + 0.5

    @property
    def xc(self):
        """obj:`np.ndarray`: Matrix of shape m x n equal to cell image with x coordinates on p(x)"""
        return self.calc_xc(self.x_coords, self.y_coords)

    @property
    def yc(self):
        """obj:`np.ndarray`: Matrix of shape m x n equal to cell image with y coordinates on p(x)"""
        return self.p(self.xc)

    @property
    def rc(self):
        return self.calc_rc(self.x_coords, self.y_coords)

    @property
    def psi(self):
        psi_rad = np.arcsin(np.divide(self.y_coords - self.yc, self.rc))
        return psi_rad * (180/np.pi)

    def p(self, x_arr):
        """
        Function to call the polynomial describing the cell spine

        Parameters
        ----------

        x_arr : obj:`np.ndarray`
            Input x array

        Returns
        -------
        obj: `np.ndarray`
            Output array of shape equal to input array with values p(x)

        """

        a0, a1, a2 = self.coeff
        return a0 + a1*x_arr + a2*x_arr**2

    def p_dx(self, x_arr):
        """
        Function to call the first derivative of the polynomial describing the cell spine to coordinate x

        Parameters
        ----------

        x_arr : obj:`np.ndarray`
            Input x array

        Returns
        -------
        obj: `np.ndarray`
            Output array of shape equal to input array with values p'(x)

        """

        a0, a1, a2 = self.coeff
        return a1 + 2 * a2 * x_arr

    def transform(self, x, y, src='cart', tgt='mpl'):
        raise DeprecationWarning('uhohhhh')

        if src == 'cart':
            xt1 = x
            yt1 = y
        elif src == 'mpl':
            xt1 = x
            yt1 = self.shape[0] - y - 0.5
        elif src == 'matrix':
            yt1 = self.shape[0] - x - 0.5
            xt1 = y + 0.5
        else:
            raise ValueError("Invalid source coordinates")

        if tgt == 'cart':
            xt2 = xt1
            yt2 = yt1
        elif tgt == 'mpl':
            xt2 = xt1
            yt2 = self.shape[0] - yt1 - 0.5#!!
        elif tgt == 'matrix':
            xt2 = self.shape[0] - yt1 - 0.5
            yt2 = xt1 - 0.5
        else:
            raise ValueError("Invalid target coordinates")
        return xt2, yt2

    def q(self, x, xp):
        """returns q(x) where q(x) is the line perpendicular to p(x) at xp"""
        return (-x / self.p_dx(xp)) + self.p(xp) + (xp / self.p_dx(xp))

    @staticmethod
    def _initial_guesses(data):
        if data.binary_img is not None:
            r = np.sqrt(mh.distance(data.binary_img).max())
            area = np.sum(data.binary_img)
            l = (area - np.pi*r**2) / (2*r)
            y_cen, x_cen = mh.center_of_mass(data.binary_img)
            xl, xr = x_cen - l/2, x_cen + l/2
            coeff = np.array([y_cen, 0.01, 0.0001])

        elif data.storm_data:
            NotImplementedError("Optimization based on only storm data is not implemented")

        return xl, xr, r, coeff

    def get_core_points(self, xl=None, xr=None):
        xl = xl if xl else self.xl
        xr = xr if xr else self.xr

        im_x, im_y = np.nonzero(self.data.data_dict['binary'])
        x_range = np.arange(int(xl), int(xr))
        y = np.array([np.nanmean(np.where(im_y == y, im_x, np.nan)) for y in x_range])

        return x_range, y

    def _initial_fit(self):
        x, y = self.get_core_points()
        return np.polyfit(x, y, 2)[::-1]


def worker(obj, **kwargs):
    return obj.optimize(**kwargs)

def worker_pb(pbar, obj, **kwargs):
    res = obj.optimize(**kwargs)
    pbar.update()
    return res

import contextlib

class DummyTqdmFile(object):
    """Dummy file-like that will write to tqdm"""
    file = None
    def __init__(self, file):
        self.file = file

    def write(self, x):
        # Avoid print() second call (useless \n)
        if len(x.rstrip()) > 0:
            tqdm.write(x, file=self.file)

    def flush(self):
        return getattr(self.file, "flush", lambda: None)()

@contextlib.contextmanager
def std_out_err_redirect_tqdm():
    orig_out_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = map(DummyTqdmFile, orig_out_err)
        yield orig_out_err[0]
    # Relay exceptions
    except Exception as exc:
        raise exc
    # Always restore sys.stdout/err if necessary
    finally:
        sys.stdout, sys.stderr = orig_out_err



class CellList(object):

    def optimize(self, data_name='binary', objective=None, **kwargs):  #todo refactor dclass to data_src or data_name
        #todo threaded and stuff
        for c in tqdm(self):
            c.optimize(data_name=data_name, objective=objective, **kwargs)

    def optimize_mp(self, data_name='binary', objective=None, processes=None, pbar=True, **kwargs):
        kwargs = {'data_name': data_name, 'objective': objective, **kwargs}
        pool = mp.Pool(processes=processes)

        f = partial(worker, **kwargs)
        # pb = tqdm(total=len(self))
        # f = partial(worker_pb, pb, **kwargs) if pbar else partial(worker, **kwargs)

      #  res = pool.map(f, self)


        #res = list(tqdm(pool.imap(f, self), total=len(self))) # works but doenst update in real time

        res = []
        with std_out_err_redirect_tqdm() as orig_stdout:
            with tqdm(total=len(self), file=orig_stdout, position=0) as pbar:
                for i, r in tqdm(enumerate(pool.imap(f, self))):
                    pbar.update(1)
                    res.append(r)

        pool.close()
        pool.join()

        for (r, v), cell in zip(res, self):
            cell.coords.sub_par(r)

    def execute_mp(self, worker, processes=None):
        pool = mp.Pool(processes=processes)
        res = pool.map(worker, self)

        pool.close()
        pool.join()

        return res

    def execute(self, worker):
        res = map(worker, self)

        return res

    def append(self, cell_obj):
        assert isinstance(cell_obj, Cell)
        self.cell_list.append(cell_obj)

    def __init__(self, cell_list):
        self.cell_list = np.array(cell_list)

    def __len__(self):
        return self.cell_list.__len__()

    def __iter__(self):
        return self.cell_list.__iter__()

    def __getitem__(self, key):
        if type(key) == int:
            return self.cell_list.__getitem__(key)
        else:
            return CellList(self.cell_list.__getitem__(key))


        # #todo what about boolean array indices slicing?
        # if type(key) == slice:
        #     return CellList(self.cell_list.__getitem__(key))
        # else:
        #     return self.cell_list.__getitem__(key)

    def __setitem__(self, key, value):
        self.cell_list.__setitem__(key, value)

    def __delitem__(self, key):
        self.cell_list.__delitem__(key)

    def __reversed__(self):
        return self.cell_list.__reversed__()

    def __contains__(self, item):
        return self.cell_list.__contains__(item)

    def r_dist(self, stop, step, data_name='', norm_x=False, storm_weight='points'):
        numpoints = len(np.arange(0, stop+step, step))
        out_arr = np.zeros((len(self), numpoints))
        for i, c in enumerate(self):
            x, y = c.r_dist(stop, step, data_name=data_name, norm_x=norm_x, storm_weight=storm_weight)
            out_arr[i] = y

        return x, out_arr

    def l_dist(self, stop, step, data_name='', norm_x=False, storm_weight='points'):
        raise NotImplementedError()

    def a_dist(self):
        raise NotImplementedError()

    @property
    def radius(self):
        return np.array([c.radius for c in self])

    @property
    def length(self):
        return np.array([c.length for c in self])

    @property
    def area(self):
        return np.array([c.area for c in self])

    @property
    def volume(self):
        return np.array([c.volume for c in self])

    @property
    def label(self):
        return np.array([c.label for c in self])

    @property
    def name(self):
        return np.array([c.name for c in self])

    def get_intensity(self, mask='binary', data_name='') -> np.ndarray:
        return np.array([c.get_intensity(mask=mask, data_name=data_name) for c in self])

    def copy(self):
        return CellList([cell.copy() for cell in self])



#http://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
#@Vladimir Ignatyev @Greenstick
def printProgress(iteration, total, prefix='', suffix='', decimals=1, barLength=50):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        barLength   - Optional  : character length of bar (Int)
    """
    formatStr = "{0:." + str(decimals) + "f}"
    percents = formatStr.format(100 * (iteration / float(total)))
    filledLength = int(round(barLength * iteration / float(total)))
    bar = 'X' * filledLength + '-' * (barLength - filledLength)
    line = '\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)
    #sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stdout.write(line)
    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()
