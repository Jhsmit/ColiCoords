#https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.basinhopping.html#scipy.optimize.basinhopping

import mahotas as mh
import math
import numpy as np
from scipy.ndimage.interpolation import rotate as scipy_rotate
from symfit import Fit

from cellcoordinates.data_models import Data
from cellcoordinates.optimizers import STORMOptimizer, BinaryOptimizer


# todo obj or class? in docstring
class Cell(object):
    """Main cell object
    
    Attributes
    ----------
    data : obj:`Data`
        Class containing all data
    coords: obj:`Coordinates` 
        Class describing cell's coordinate system and associated functions
    #todo perhaps move this shizzle to a dedicated metadataclass
    img_name : str:
        Name of the image from which the cell originated
    label : int:
        cell number in the original labeled binary
        
    
    Parameters
    ----------
    bf_img : :obj:`np.ndarray`, optional
        Numpy array of shape MxN with the cell's binary image
    binary_img : :obj:`np.ndarray`, optional 
        Numpy array of shape MxN with the cell's binary image
    fl_data : dict, optional
        Fluorescence data either dict, np.npdarray (or directly the corresponding objects)
    storm_data : :obj:`np.ndarray`
        Storm coordinates (x, y) in ImageJ coordinates
    label : str,
        Label identifying the cell object


    --> name: name referring original image
    --> idx: index of the cell in the labelled binary
    --> label: property returning {}{}.format(name, idx) or sth
    """

    def __init__(self, bf_img=None, binary_img=None, flu_data=None, storm_table=None, *args, **kwargs):
        if 'data_dict' in kwargs:
            data_dict = kwargs['data_dict']
            bf_img = data_dict.pop('Brightfield', None)
            binary_img = data_dict.pop('Binary', None)
            storm_table = data_dict.pop('STORMTable', None)
            flu_data = data_dict
            self.data = Data()
            self.data.add_datasets(bf_img=bf_img, binary_img=binary_img, flu_data=flu_data, storm_table=storm_table)
        elif 'data_obj' in kwargs:
            self.data = kwargs['data_obj']
        else:
            self.data = Data()
            self.data.add_datasets(bf_img=bf_img, binary_img=binary_img, flu_data=flu_data, storm_table=storm_table)

        self.coords = Coordinates(self.data)
        if 'label' in kwargs:
            print('move labels to global metadata')
            pass

    def optimize(self, dclass=None, method='photons', verbose=True):
        if not dclass:
            if self.data.binary_img is not None and self.data.storm_table is None:
                optimizer = BinaryOptimizer(self)
            elif not self.data.binary_img and self.data.storm_table:
                optimizer = STORMOptimizer(self, method=method)
            elif self.data.binary_img and self.data.storm_table:
                raise ValueError("Please specify optimize method")
            elif not self.data.binary_img and not self.data.storm_table:
                raise ValueError("Please specify optimize method")

        elif dclass == 'Binary':
            optimizer = BinaryOptimizer(self)
        elif dclass == 'Fluorescence':
            raise NotImplementedError
            #optimizer = FluorescenceOptimizer(self)
        elif dclass == 'STORM':
            optimizer = STORMOptimizer(self, method=method)
        else:
            raise ValueError("Invalid value for optimize_method")

        #todo optimizer as property
        #optimizer.execute()
        optimizer.optimize_overall(verbose=verbose)

    @property
    def length(self):
        """float: Length of the cell obtained by integration of the spine arc length from xl to xr"""
        a0, a1, a2 = self.coords.coeff
        xr, xl = self.coords.xl, self.coords.xr
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

    @property
    def label(self):
        try:
            return '{}{}'.format(self.name, str(self.label).zfill(2))
        except AttributeError:
            return None


    #todo choose fluorescence channel or storm
    def radial_distribution(self, stop, step, src=''):
        def bin_func(r, flu, bins):
            r_flat = r.flatten()
            i_sort = r_flat.argsort()
            r_sorted = r_flat[i_sort]
            flu_sorted = flu.flatten()[i_sort]

            bin_inds = np.digitize(r_sorted,
                                   bins) - 1  # -1 to assure points between 0 and step are in bin 0 (the first)
            yvals = np.bincount(bin_inds, weights=flu_sorted, minlength=len(bins)) / np.bincount(bin_inds, minlength=len(bins))
            return yvals

        bins = np.arange(0, stop+step, step)
        xvals = bins + 0.5 * step  # xval is the middle of the bin

        if not src:
            data = list(self.data.flu_dict.values())[0] #yuck
        elif src == 'storm':
            raise NotImplementedError('Calculating radial distribution of STORM data not yet implemented')
        else:
            try:
                data = self.data.data_dict[src]
            except KeyError:
                raise ValueError('Chosen data not found')

        if data.ndim == 2:
            yvals = bin_func(self.coords.rc, data, bins)
        elif data.ndim == 3:
            yvals = np.vstack([bin_func(self.coords.rc, d, bins) for d in data])
        else:
            raise ValueError('Invalid fluorescence image dimensions')

        return xvals, yvals


class Coordinates(object):
    """Cell's coordinate system described by the polynomial f(x) and associated functions

    Attributes
    ----------
    xl : float
        cartesian x-coordinate of cell left pole 
    xr : float
        cartesian x-coordinate of cell right pole
    r : float
        radius of the cell (half the short axis width)
    coeff : obj:`np.ndarray`
        polynomial f(x) = a0 + a1*x + a2*x**2 coefficients a0, a1 and a2 
    
    """
    def __init__(self, data):
        self.coeff = np.array([1., 1., 1.])
        self.xl, self.xr, self.r, self.coeff = self._initial_guesses(data) #todo implement
        self.shape = data.shape

    def calc_xc(self, xp, yp):
        """ Calculates the coordinate xc on f(x) closest to x, y 
        
        All coordinates are cartesian. Solutions are found by solving the cubic equation.
        https://en.wikipedia.org/wiki/Cubic_function#Algebraic_solution
        """
        #todo check if this function allows 2d arrays (it should)

        a0, a1, a2 = self.coeff
        # Converting of cell spine polynomial coefficients to coefficients of polynomial giving distance r
        a, b, c, d = 4*a2**2, 6*a1*a2, 4*a0*a2 + 2*a1**2 - 4*a2*yp + 2, 2*a0*a1 - 2*a1*yp - 2*xp
        #a: float, b: float, c: array, d: array
        discr = 18*a*b*c*d - 4*b**3*d + b**2*c**2 - 4*a*c**3 - 27*a**2*d**2

        def solve_general(a, b, c, d):
            """
            Solve cubic polynomial in the form a*x^3 + b*x^2 + c*x + d
            Only works if polynomial discriminant < 0, then there is only one real root which is the one that is returned.
            https://en.wikipedia.org/wiki/Cubic_function#General_formula
            :return (float): Only real root
            """
            d0 = b ** 2. - 3. * a * c
            d1 = 2. * b ** 3. - 9. * a * b * c + 27. * a ** 2. * d
            dc = np.cbrt((d1 + np.sqrt(d1 ** 2. - 4. * d0 ** 3.)) / 2.)
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

        return x_c

    def calc_rc(self, xp, yp):
        """
        Applies endcap limits xl and xr and calculates distance r to cell spine
        :param xp: 1D array of x coordinates
        :param yp: 1D array of y coordinates
        :return: 1D array of distances r from (x, y) to (xc, p(xc))
        """
        xc = self.calc_xc(xp, yp)
        xc[xc < self.xl] = self.xl
        xc[xc > self.xr] = self.xr

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
        return np.repeat(np.arange(ymax), xmax).reshape(ymax, xmax)[::-1, :] + 0.5

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
        return a1 + a2 * x_arr

    def transform(self, x, y, src='cart', tgt='mpl'):
        if src == 'cart':
            xt1 = x
            yt1 = y
        elif src == 'mpl':
            xt1 = x
            yt1 = self.shape[0] - y
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
            yt2 = self.shape[0] - yt1
        elif tgt == 'matrix':
            xt2 = self.shape[0] - yt1 - 0.5
            yt2 = xt1 - 0.5
        else:
            raise ValueError("Invalid target coordinates")
        return xt2, yt2

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


class CellList(object):
    def __init__(self, cell_list):
        self.cell_list = cell_list

    def __len__(self):
        return self.cell_list.__len__()

    def __iter__(self):
        return self.cell_list.__iter__()

    def __getitem__(self, key):
        return self.cell_list.__getitem__(key)

    def __setitem__(self, key, value):
        self.cell_list.__setitem__(key, value)

    def __delitem__(self, key):
        self.cell_list.__delitem__(key)

    def __reversed__(self):
        return self.cell_list.__reversed__()

    def __contains__(self, item):
        return self.cell_list.__contains__(item)

    @property
    def length(self):
        return np.array([c.length for c in self])

    @property
    def area(self):
        return np.array([c.area for c in self])

    @property
    def volume(self):
        return np.array([c.volume for c in self])