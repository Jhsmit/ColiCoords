#https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.basinhopping.html#scipy.optimize.basinhopping

import mahotas as mh
import numpy as np
import operator
from functools import partial
from colicoords.optimizers import Optimizer
from colicoords.support import allow_scalars
from scipy.integrate import quad
from scipy.optimize import fsolve
#import multiprocessing as mp
import multiprocess as mp
from tqdm import tqdm
import sys
import contextlib


class Cell(object):
    """ ColiCoords' main single-cell object.

    This class hold all single-cell associated data as well as an internal coordinate system.

    Attributes:
        data (:class:`Data`): Holds all data describing this single cell.
        coords (:class:`Coordinates`): Calculates and optimizes the cell's coordinate system.
        name (:obj:`str`): Name identifying the cell (optional)
        index (:obj:`int`) Index of the cell in a cell list (for maintaining order upon load/save)
    """

    def __init__(self, data_object, name=None, **kwargs):
        """
        Args:
            data_object (:class:`Data`): Data class holding all data which describes this single cell
            name (:obj:`str`): Name to identify this single cell.
                #todo generate names when trying to save cell_list without names to disk
        """

        self.data = data_object
        self.coords = Coordinates(self.data)
        self.name = name
        self.index = kwargs.pop('index', None)

    def optimize(self, data_name='binary', objective=None, **kwargs):
        # todo find out if callable is a thing
        """ Optimize the cell's coordinate system. The optimization is performed on the data element given by `data_name`
            using objective function `objective`. See the documentation REF or colicoords.optimizers for more details.

        Args:
            data_name (:obj:`str`): Name of the data element on which coordinate optimization is performed.
            objective (:obj:`str` or :obj:`callable`):
            **kwargs: keyword arguments which are passed to `Optimizer.optimize`
        """
        optimizer = Optimizer(self, data_name=data_name, objective=objective)
        return optimizer.optimize(**kwargs)

    @property
    def radius(self):
        """:obj:`float`: Radius of the cell in pixels"""
        return self.coords.r

    @property
    def length(self):
        """:obj:`float`: Length of the cell in pixels. Obtained by integration of the spine arc length from `xl` to `xr`"""
        a0, a1, a2 = self.coords.coeff
        xl, xr = self.coords.xl, self.coords.xr
        l = (1 / (4 * a2)) * (
            ((a1 + 2 * a2 * xr) * np.sqrt(1 + (a1 + 2 * a2 * xr) ** 2) + np.arcsinh((a1 + 2 * a2 * xr))) -
            ((a1 + 2 * a2 * xl) * np.sqrt(1 + (a1 + 2 * a2 * xl) ** 2) + np.arcsinh((a1 + 2 * a2 * xl)))
        )

        return l

    @property
    def circumference(self):
        """:obj:`float`: Circumference of the cell in pixels"""
        #http://tutorial.math.lamar.edu/Classes/CalcII/ParaArcLength.aspx
        def integrant_top(t, a1, a2, r):
            return np.sqrt(1 + (a1 + 2*a2*t)**2 + ( (4*a2**2*r**2) / (1 + (a1 + 2*a2*t)**2)**2 ) + ( (4*a2*r) / np.sqrt(1 + (a1 + 2*a2*t)) ))

        def integrant_bot(t, a1, a2, r):
            return np.sqrt(1 + (a1 + 2 * a2 * t) ** 2 + ((4 * a2 ** 2 * r ** 2) / (1 + (a1 + 2 * a2 * t) ** 2) ** 2) - ((4 * a2 * r) / np.sqrt(1 + (a1 + 2 * a2 * t))))

        top, terr = quad(integrant_top, self.coords.xl, self.coords.xr, args=(self.coords.a1, self.coords.a2, self.coords.r))
        bot, berr = quad(integrant_bot, self.coords.xl, self.coords.xr, args=(self.coords.a1, self.coords.a2, self.coords.r))

        return top + bot + 2*np.pi*self.coords.r

    @property
    def area(self):
        """:obj:`float`: Area (2d) of the cell in square pixels"""
        return 2*self.length*self.coords.r + np.pi*self.coords.r**2

    @property
    def surface(self):
        """:obj:`float`: Total surface area (3d) of the cell in square pixels"""
        return self.length*2*np.pi*self.coords.r + 4*np.pi*self.coords.r**2

    @property
    def volume(self):
        """:obj:`float`: Volume of the cell in cubic pixels"""
        return np.pi*self.coords.r**2*self.length + (4/3)*np.pi*self.coords.r**3

    def a_dist(self):
        raise NotImplementedError()

    def l_dist(self, nbins, data_name='', norm_x=False, r_max=None, storm_weight=False):
        """Calculated the longitudinal distribution of a given data element.

        Args:
            nbins (:obj:`int`): Number of bins between xl and xr
            data_name (:obj:`str`): Name of the data element to use.
            norm_x (:obj:`bool`): If *True* the output distribution will be normalized.
            r_max: (:obj:`float`): Datapoints within r_max from the cell midline will be included. If *None* the value
                from the cell's coordinate system will be used.
            storm_weight: If *True* the datapoints of the specified STORM-type data will be weighted by their intensity.

        Returns:
            :obj:`tuple`: tuple containing:

                xvals (:class:`~numpy.ndarray`) Array of distances along the cell midline, values are the middle of the bins

                yvals (:class:`~numpy.ndarray`) Array of in bin heights

        """
        #todo check the bins
        if not data_name:
            data_elem = list(self.data.flu_dict.values())[0] #yuck
        else:
            try:
                data_elem = self.data.data_dict[data_name]
            except KeyError:
                raise ValueError('Chosen data not found')

        r_max = r_max if r_max else self.coords.r
        stop = 1 if norm_x else self.length
        bins = np.linspace(0, stop, num=nbins, endpoint=False)
        xvals = bins + 0.5 * np.diff(bins)[0]  # xval is the middle of the bin

        if data_elem.ndim == 1:
            assert data_elem.dclass == 'storm'
            x = data_elem['x']
            y = data_elem['y']

            r = self.coords.calc_rc(x, y)
            xc = self.coords.calc_xc(x, y)

        elif data_elem.ndim == 2 or data_elem.ndim == 3:  # image data
            r = self.coords.rc
            xc = self.coords.xc

        else:
            raise ValueError('Invalid data element dimensions')

        b1 = r < r_max
        b2 = np.logical_and(xc >= self.coords.xl, xc <= self.coords.xr)
        b = np.logical_and(b1, b2)
        x_len = _calc_len(self.coords.xl, xc[b].flatten(), self.coords.coeff)
        x_len = x_len / self.length if norm_x else x_len

        if data_elem.ndim == 1:
            y_weight = data_elem['intensity'][b] if storm_weight else None
            yvals = self._bin_func(x_len, y_weight, bins)

        elif data_elem.ndim == 2:
            y_weight = np.clip(data_elem[b].flatten(), 0, None)
            yvals = self._bin_func(x_len, y_weight, bins)

        elif data_elem.ndim == 3:
            yvals = np.array([self._bin_func(x_len, y_weight[b].flatten(), bins) for y_weight in data_elem])

        return xvals, yvals

    def l_classify(self, data_name=''):
        """Classifies foci in STORM-type data by they x-position along the long axis.

        The spots are classfied into 3 categories: 'poles', 'between' and 'mid'. The pole category are spots who are to
            the left and right of xl and xr, respectively. The class 'mid' is a section in the middle of the cell with a
            total length of half the cell's length, the class 'between' is the remaining two quarters between 'mid' and
            'poles'

        Args:
            data_name (:obj:`str`): Name of the STORM-type data element to classifty. When its not specified the first
                STORM data element is used.

        Returns:
            :obj:`tuple`: Tuple with number of spots in poles, between and mid classes, respectively.
        """
        if not data_name:
            data_elem = list(self.data.storm_dict.values())[0]
        else:
            data_elem = self.data.data_dict[data_name]
            assert data_elem.dclass == 'storm'

        x, y = data_elem['x'], data_elem['y']
        lc = self.coords.calc_lc(x, y)
        lq1 = self.length / 4
        lq3 = 3*lq1

        poles = np.sum(lc <= 0) + np.sum(lc >= self.length)
        between = np.sum(np.logical_and(lc > 0, lc < lq1)) + np.sum(np.logical_and(lc < self.length, lc > lq3))
        mid = np.sum(np.logical_and(lc >= lq1, lc <= lq3))

        try:
            assert len(x) == (poles + between + mid)
        except AssertionError:
            raise ValueError("Invalid number of points")

        return poles, between, mid

    def r_dist(self, stop, step, data_name='', norm_x=False, xlim=None, storm_weight=False):
        #todo test xlim!
        """ Calculates the radial distribution of a given data element.

        Args:
            stop (:obj:`float`): Until how far from the cell spine the radial distribution should be calculated
            step (:obj:`float`): The binsize of the returned radial distribution
            data_name (:obj:`str`): The name of the data element on which to calculate the radial distribution
            norm_x (:obj:`bool`): If `True` the returned distribution will be normalized with the cell's radius set to 1.
            xlim (:obj:`str`): If `None`, all datapoints are taking into account. This can be limited by providing the
                value `full` (omit poles only), 'poles' (include only poles), or a float value which will limit the data
                points with around the midline where xmid - xlim < x < xmid + xlim.
            storm_weight (:obj:`bool`): Only applicable for analyzing STORM-type data elements. If `True` the returned
                histogram is weighted with the number of photons measured.

        Returns:
            :obj:`tuple`: tuple containing:

                xvals (:class:`~numpy.ndarray`) Array of distances from the cell midline, values are the middle of the bins

                yvals (:class:`~numpy.ndarray`) Array of in bin heights
        """

        #todo this nest of if else's needs some cleanup, the xlim clause appears thrice!

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
                    b = ((xc > self.coords.xl) * (xc < self.coords.xr)).astype(bool)
                elif xlim == 'poles':
                    b = ((xc <= self.coords.xl) * (xc >= self.coords.xr)).astype(bool)
                else:
                    mid_x = (self.coords.xl + self.coords.xr) / 2
                    b = (xc > mid_x - xlim) * (xc < mid_x + xlim).astype(bool)

                bin_r = r[b].flatten()
            else:
                bin_r = r.flatten()
                b = np.ones_like(x).astype(bool)

            y_weight = data_elem['intensity'][b] if storm_weight else None
            yvals = self._bin_func(bin_r, y_weight, bins)

        elif data_elem.ndim == 2:
            r = self.coords.rc / self.coords.r if norm_x else self.coords.rc
            if xlim:
                if xlim == 'full':
                    b = (self.coords.xc > self.coords.xl) * (self.coords.xc < self.coords.xr).astype(bool)
                elif xlim == 'poles':
                    b = ((self.coords.xc <= self.coords.xl) * (self.coords.xc >= self.coords.xr)).astype(bool)
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
            if xlim:
                if xlim == 'full':
                    b = (self.coords.xc > self.coords.xl) * (self.coords.xc < self.coords.xr).astype(bool)
                elif xlim == 'poles':
                    b = ((self.coords.xc <= self.coords.xl) * (self.coords.xc >= self.coords.xr)).astype(bool)
                else:
                    mid_x = (self.coords.xl + self.coords.xr)/2
                    b = (self.coords.xc > mid_x - xlim)*(self.coords.xc < mid_x + xlim).astype(bool)
            else:
                b = True

            r = self.coords.rc / self.coords.r if norm_x else self.coords.rc
            yvals = np.vstack([self._bin_func(r[b].flatten(), d[b].flatten(), bins) for d in data_elem])
        else:
            raise ValueError('Invalid data element dimensions')
        return xvals, yvals

    def measure_r(self, data_name='brightfield', in_place=True):
        """
        Measure the radius of the cell by finding the intensity-midpoint of the radial distribution derived from
        brightfield (default) or another data element.

        Args:
            data_name (:obj:`str`): Name of the data element to use.
            in_place (:obj:`bool`): If `True` the found value of `r` is directly substituted in the cell's coordinate
                system, otherwise the value is returned.

        Returns:
            The measured radius `r` if `in_place` is `False`, otherwise `None`.
        """
        x, y = self.r_dist(15, 1, data_name=data_name)  # todo again need sensible default for stop
        mid_val = (np.min(y) + np.max(y)) / 2

        imin = np.argmin(y)
        imax = np.argmax(y)
        try:
            r = np.interp(mid_val, y[imin:imax], x[imin:imax])
            if in_place:
                self.coords.r = r
            else:
                return r

        except ValueError:
            print('r value not found')

    def reconstruct_cell(self, data_name, norm_x=False, r_scale=1, **kwargs):
        #todo stop and step defaults when norm_x=True?
        #todo allow reconstruction of standardized cell shape
        """
            Reconstruct the cell from a given data element and the cell's current coordinate system.
        Args:
            data_name (:obj:`str`): Name of the data element to use
            norm_x (:obj:`bool`): Boolean indicating whether or not to normalize to r=1
            r_scale:
            **kwargs: kwargs to optionally provide 'stop' and 'step' values for the used `r_dist`.

        Returns:
            :class:`~numpy.ndarray`: Image of the reconstructed cell
        """

        stop = kwargs.pop('stop', np.ceil(np.max(self.data.shape)/2))
        step = kwargs.pop('step', 1)

        xp, fp = self.r_dist(stop, step, data_name=data_name, norm_x=norm_x)
        interp = np.interp(r_scale*self.coords.rc, xp, np.nan_to_num(fp))  #todo check nantonum cruciality

        return interp

    def get_intensity(self, mask='binary', data_name=''):
        """ Returns the mean fluorescence intensity in the region masked by either the binary image or synthetic
            binary image derived from the cell's coordinate system

        Args:
            mask (:obj:`str`): Either 'binary' or 'coords' to specify the source of the mask used.
                'binary' uses the binary imagea as mask, 'coords' uses reconstructed binary from coordinate system
            data_name (:obj:`str`): The name of the image data element to get the intensity values from.

        Returns:
            :obj:`float`: Mean fluorescence pixel value

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

    @staticmethod
    def _bin_func(xvals, y_weight, bins):
        """bins xvals in given bins using y_weight as weights"""
        i_sort = xvals.argsort()
        r_sorted = xvals[i_sort]
        y_weight = y_weight[i_sort] if y_weight is not None else y_weight
        bin_inds = np.digitize(r_sorted,
                               bins) - 1  # -1 to assure points between 0 and step are in bin 0 (the first)
        yvals = np.bincount(bin_inds, weights=y_weight, minlength=len(bins))
        if y_weight is not None:
            yvals /= np.bincount(bin_inds, minlength=len(bins))
        return np.nan_to_num(yvals)

    def copy(self):
        """
        Make a copy of the cell object and all its associated data elements

        Returns:
            :class:`Cell`: Copied cell object

        """
        #todo needs testing (this is done?) arent there more properties to copy?
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

    def __init__(self, data, initialize=True, **kwargs):
        """
        Args:
            data (:class:`Data`): Data object holding all the cell's data elements
            initialize (:obj:`bool`): If `True` the cell coordinates will be initialized by deriving initial guesses
                from the binary image.
            **kwargs: optional keyword arguments. If `initialize` is `False` parameters can be passed in `kwargs` to
                manually set these parameters.
        """
        self.data = data
        self.coeff = np.array([1., 1., 1.])

        if initialize:
            self.xl, self.xr, self.r, self.coeff = self._initial_guesses(data) #refactor to class method
            self.coeff = self._initial_fit()
            self.shape = data.shape
        else:
            for p in self.parameters + ['shape']:
                setattr(self, p, kwargs.pop(p, None))

    @property
    def a0(self):
        """float: Polynomial p(x) 0th degree coefficient"""
        return self.coeff[0]

    @a0.setter
    def a0(self, value):
        self.coeff[0] = value

    @property
    def a1(self):
        """float: Polynomial p(x) 1st degree coefficient"""
        return self.coeff[1]

    @a1.setter
    def a1(self, value):
        self.coeff[1] = value

    @property
    def a2(self):
        """float: Polynomial p(x) 2nd degree coefficient"""
        return self.coeff[2]

    @a2.setter
    def a2(self, value):
        self.coeff[2] = value

    def sub_par(self, par_dict):
        """
        Substitute the values in `par_dict` as the coordinate systems parameters.

        Args:
            par_dict (:obj:`dict`): Dictionary with parameters which values are set to the attributes.
        """
        for k, v in par_dict.items():
            setattr(self, k, v)

    @allow_scalars
    def calc_xc(self, xp, yp):
        """ Calculates the coordinate xc on p(x) closest to xp, yp
        
        All coordinates are cartesian. Solutions are found by solving the cubic equation.

        Args:
            xp (:`obj`:float: or :class:`~numpy.ndarray`:): Input scalar or vector/matrix x-coordinate. Must be the same shape as yp
            yp (:`obj`:float: or :class:`~numpy.ndarray`:): Input scalar or vector/matrix x-coordinate. Must be the same shape as xp

        Returns:
            :`obj`:float: or :class:`~numpy.ndarray`: Cellular x-coordinate for point(s) xp, yp
        """

        assert xp.shape == yp.shape
        #https://en.wikipedia.org/wiki/Cubic_function#Algebraic_solution
        a0, a1, a2 = self.coeff
        #xp, yp = xp.astype('float32'), yp.astype('float32')
        # Converting of cell spine polynomial coefficients to coefficients of polynomial giving distance r
        a, b, c, d = 4*a2**2, 6*a1*a2, 4*a0*a2 + 2*a1**2 - 4*a2*yp + 2, 2*a0*a1 - 2*a1*yp - 2*xp
        #a: float, b: float, c: array, d: array
        discr = 18*a*b*c*d - 4*b**3*d + b**2*c**2 - 4*a*c**3 - 27*a**2*d**2

        if np.any(discr == 0):
            raise ValueError('Discriminant equal to zero encountered. This has never happened before! What did you do?')

        if np.all(discr < 0):
            x_c = _solve_general(a, b, c, d)
        else:
            x_c = np.zeros(xp.shape)
            mask = discr < 0

            general_part = _solve_general(a, b, c[mask], d[mask])
            trig_part = solve_trig(a, b, c[~mask], d[~mask])

            x_c[mask] = general_part
            x_c[~mask] = trig_part

        return x_c

    @allow_scalars
    def calc_xc_mask(self, xp, yp):
        """ Calculated whether point (xp, yp) is in either the left or right polar areas, or in between.

        Returned values are 1 for left pole, 2 for middle, 3 for right pole.

        Args:
            xp (:`obj`:float: or :class:`~numpy.ndarray`:): Input scalar or vector/matrix x-coordinate. Must be the same shape as yp
            yp (:`obj`:float: or :class:`~numpy.ndarray`:): Input scalar or vector/matrix x-coordinate. Must be the same shape as xp

        Returns:
            :`obj`:float: or :class:`~numpy.ndarray`: Array to mask different cellular regions.
        """

        idx_left, idx_right, xc = self.get_idx_xc(xp, yp)
        mask = 2*np.ones_like(xp)
        xc[idx_left] = 1
        xc[idx_right] = 3

        return mask

    @allow_scalars
    def calc_xc_masked(self, xp, yp):
        """ Calculates the coordinate xc on p(x) closest to (xp, yp), where xl < xc < xr

        Args:
            xp (:`obj`:float: or :class:`~numpy.ndarray`:): Input scalar or vector/matrix x-coordinate. Must be the same shape as yp
            yp (:`obj`:float: or :class:`~numpy.ndarray`:): Input scalar or vector/matrix x-coordinate. Must be the same shape as xp

        Returns:
            :`obj`:float: or :class:`~numpy.ndarray`: Cellular x-coordinate for point(s) xp, yp, where xl < xc < xr
        """
        idx_left, idx_right, xc = self.get_idx_xc(xp, yp)
        xc[idx_left] = self.xl
        xc[idx_right] = self.xr

        return xc

    @allow_scalars
    def calc_rc(self, xp, yp):
        """ Calculates the distance of (xp, yp) to (xc, p(xc)).

        The returned value is the distance from the points (xp, yp) to the midline of the cell.

        Args:
            xp (:`obj`:float: or :class:`~numpy.ndarray`:): Input scalar or vector/matrix x-coordinate. Must be the same shape as yp
            yp (:`obj`:float: or :class:`~numpy.ndarray`:): Input scalar or vector/matrix x-coordinate. Must be the same shape as xp

        Returns:
            :`obj`:float: or :class:`~numpy.ndarray`: Distance to the midline of the cell.
        """

        xc = self.calc_xc_masked(xp, yp)
        a0, a1, a2 = self.coeff
        return np.sqrt((xc - xp)**2 + (a0 + xc*(a1 + a2*xc) - yp)**2)

    @allow_scalars
    def calc_lc(self, xp, yp):
        """ Calculates distance of xc along the midline the cell corresponding to the points (xp, yp).

        The returned value is the distance from the points (xp, yp) to the midline of the cell.

        Args:
            xp (:`obj`:float: or :class:`~numpy.ndarray`:): Input scalar or vector/matrix x-coordinate. Must be the same shape as yp
            yp (:`obj`:float: or :class:`~numpy.ndarray`:): Input scalar or vector/matrix x-coordinate. Must be the same shape as xp

        Returns:
            :`obj`:float: or :class:`~numpy.ndarray`: Distance along the midline of the cell.
        """

        xc = self.calc_xc_masked(xp, yp)
        return _calc_len(self.xl, xc, self.coeff)

    @allow_scalars
    def calc_psi(self, xp, yp):
        """ Calculates the angle between the line perpendical to the cell midline and the line between (xp, yp) and (xc, p(xc).

        The returned values are in degrees. The angle is defined to be 0 degrees for values in the upper half of the image
        (yp < p(xp)), running from 180 to zero along the right polar region, 180 degrees in the lower half and running back to
        0 degrees along the left polar region.

        Args:
            xp (:`obj`:float: or :class:`~numpy.ndarray`:): Input scalar or vector/matrix x-coordinate. Must be the same shape as yp
            yp (:`obj`:float: or :class:`~numpy.ndarray`:): Input scalar or vector/matrix x-coordinate. Must be the same shape as xp

        Returns:
            :`obj`:float: or :class:`~numpy.ndarray`: Angle psi for (xp, yp).
        """

        idx_left, idx_right, xc = self.get_idx_xc(xp, yp)
        xc[idx_left] = self.xl
        xc[idx_right] = self.xr
        yc = self.p(xc)

        psi = np.empty(xp.shape)
        #todo this no worky probably
        top = yp < self.p(xp)
        psi[top] = 0
        psi[~top] = np.pi

        th1 = np.arctan2(yp - yc, xc - xp)
        th2 = np.arctan(self.p_dx(xc))
        thetha = th1 + th2 + np.pi / 2
        psi[idx_right] = (np.pi - thetha[idx_right]) % np.pi
        psi[idx_left] = thetha[idx_left]

        return psi*(180/np.pi)

    def get_idx_xc(self, xp, yp):
        """ Finds the indices of the arrays xp an yp where they either belong to the left or right polar regins, as well as
            coordinates xc

        Args:
            xp (:`obj`:float: or :class:`~numpy.ndarray`:): Input scalar or vector/matrix x-coordinate. Must be the same shape as yp
            yp (:`obj`:float: or :class:`~numpy.ndarray`:): Input scalar or vector/matrix x-coordinate. Must be the same shape as xp

        Returns:
            :`obj`:float: or :class:`~numpy.ndarray`: Angle psi for (xp, yp).
        """

        xc = self.calc_xc(xp, yp).copy()
        yp = self.p(xc)

        # Area left of perpendicular line at xl:
        op = operator.lt if self.p_dx(self.xl) > 0 else operator.gt
        idx_left = op(yp, self.q(xc, self.xl))

        op = operator.gt if self.p_dx(self.xr) > 0 else operator.lt
        idx_right = op(yp, self.q(xc, self.xr))

        return idx_left, idx_right, xc

    @allow_scalars
    def transform(self, xp, yp):
        """ Transforms image coordinates (xp, yp) to cell coordinates (xc, lc, rc, psi)

        Args:
            xp (:`obj`:float: or :class:`~numpy.ndarray`:): Input scalar or vector/matrix x-coordinate. Must be the same shape as yp
            yp (:`obj`:float: or :class:`~numpy.ndarray`:): Input scalar or vector/matrix x-coordinate. Must be the same shape as xp

        Returns:
            :`obj`:tuple: Tuple of cellular coordinates xc, lc, rc, psi
        """
        
        xc = self.calc_xc_masked(xp, yp)
        lc = self.calc_lc(xp, yp)
        rc = self.calc_rc(xp, yp)
        psi = self.calc_psi(xp, yp)

        return xc, lc, rc, psi

    @property
    def x_coords(self):
        """:class:`~numpy.ndarray``: Matrix of shape m x n equal to cell image with cartesian x-coordinates."""
        ymax = self.shape[0]
        xmax = self.shape[1]
        return np.repeat(np.arange(xmax), ymax).reshape(xmax, ymax).T + 0.5

    @property
    def y_coords(self):
        """:class:`~numpy.ndarray`: Matrix of shape m x n equal to cell image with cartesian y-coordinates."""
        ymax = self.shape[0]
        xmax = self.shape[1]
        return np.repeat(np.arange(ymax), xmax).reshape(ymax, xmax) + 0.5

    @property
    def xc(self):
        """:class:`~numpy.ndarray`: Matrix of shape m x n equal to cell image with x coordinates on p(x)"""
        return self.calc_xc(self.x_coords, self.y_coords)

    @property
    def yc(self):
        """:class:`~numpy.ndarray`: Matrix of shape m x n equal to cell image with y coordinates on p(x)"""
        return self.p(self.xc)

    @property
    def xc_masked(self):
        """:class:`~numpy.ndarray`: Matrix of shape m x n equal to cell image with x coordinates on p(x)
            where xl < xc < xr.
        """
        return self.calc_xc_masked(self.x_coords, self.y_coords)

    @property
    def xc_mask(self):
        """:class:`~numpy.ndarray`: Matrix of shape m x n equal to cell image where elements have values 1, 2, 3 for
            left pole, middle and right pole, respectively.
        """
        return self.calc_xc_mask(self.x_coords, self.y_coords)

    @property
    def rc(self):
        """:class:`~numpy.ndarray`: Matrix of shape m x n equal to cell with distance r to the cell midline."""
        return self.calc_rc(self.x_coords, self.y_coords)

    @property
    def lc(self):
        """:class:`~numpy.ndarray`: Matrix of shape m x n equal to cell with distance l along the cell mideline."""
        return self.calc_lc(self.x_coords, self.y_coords)

    @property
    def psi(self):
        """:class:`~numpy.ndarray`: Matrix of shape m x n equal to cell with angle psi relative to the cell midline."""
        return self.calc_psi(self.x_coords, self.y_coords)

    def p(self, x_arr):
        """
            Calculate p(x).
        Args:
            x_arr (:class:`~numpy.ndarray`): Input x values.

        Returns:
            :class:`~numpy.ndarray`: p(x)
        """
        a0, a1, a2 = self.coeff
        return a0 + a1*x_arr + a2*x_arr**2

    def p_dx(self, x_arr):
        """
            Calculate the derivative p'(x) of p(x) evaluated at x.
        Args:
            x_arr (:class:`~numpy.ndarray`): Input x values.

        Returns:
            :class:`~numpy.ndarray`: p'(x)
        """
        a0, a1, a2 = self.coeff
        return a1 + 2 * a2 * x_arr

    def q(self, x, xp):
        """returns q(x) where q(x) is the line perpendicular to p(x) at xp"""
        return (-x / self.p_dx(xp)) + self.p(xp) + (xp / self.p_dx(xp))

    def get_core_points(self, xl=None, xr=None):
        """
            Returns the coordinates of the roughly estimated 'core' points of the cell. Used for determining the
                initial guesses for the coefficients of p(x).
        Args:
            xl (:obj:`float`): starting point x of where to get the 'core' points.
            xr (:obj:`float`): end point x of where to get the 'core' points.

        Returns:
            (tuple): tuple containing:

                xvals (:class:`np.ndarray`) Array of x coordinates of 'core' points.
                yvals (:class:`np.ndarray`) Array of y coordinates of 'core' points.
        """
        xl = xl if xl else self.xl
        xr = xr if xr else self.xr

        im_x, im_y = np.nonzero(self.data.data_dict['binary'])
        x_range = np.arange(int(xl), int(xr))
        y = np.array([np.nanmean(np.where(im_y == y, im_x, np.nan)) for y in x_range])

        return x_range, y

    @staticmethod
    def _initial_guesses(data):
        if data.binary_img is not None:
            r = np.sqrt(mh.distance(data.binary_img).max())
            area = np.sum(data.binary_img)
            l = (area - np.pi*r**2) / (2*r)
            y_cen, x_cen = mh.center_of_mass(data.binary_img)
            xl, xr = x_cen - l/2, x_cen + l/2
            coeff = np.array([y_cen, 0.01, 0.0001])

        else:
            raise NotImplementedError("Binary image is required for initial guesses of cell coordinates")

        return xl, xr, r, coeff

    def _initial_fit(self):
        x, y = self.get_core_points()
        return np.polyfit(x, y, 2)[::-1]


def worker(obj, **kwargs):
    return obj.optimize(**kwargs)


def worker_pb(pbar, obj, **kwargs):
    res = obj.optimize(**kwargs)
    pbar.update()
    return res


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
    """Object holding a list of cell objects exposing several methods to either apply functions to all cells or to extract
        values from all cell objects. This object supports iteration over Cell objects and Numpy-style array indexing.
    Attributes:
        cell_list (:class:`~numpy.ndarray`): Numpy array of `Cell` objects

    """
    def __init__(self, cell_list):
        """

        Args:
            cell_list:
        """
        self.cell_list = np.array(cell_list)

    def optimize(self, data_name='binary', objective=None, **kwargs):
        #todo REF
        #todo describe objective kwargs
        """ Optimize the all the cell's coordinate system. The optimization is performed on the data element given by
            `data_name` using objective function `objective`. See the documentation REF or colicoords.optimizers for
            more details.

        Args:
            data_name (:obj:`str`): Name of the data element on which coordinate optimization is performed.
            objective (:obj:`str` or :obj:`callable`):
            **kwargs: keyword arguments which are passed to `Optimizer.optimize`
        """

        for c in tqdm(self):
            c.optimize(data_name=data_name, objective=objective, **kwargs)

    def optimize_mp(self, data_name='binary', objective=None, processes=None, pbar=True, **kwargs):
        """ Optimize all cell's coordinate systems using `optimize` through parallel computing. Note that if this
            is called the call must be protected by if __name__ == '__main__'.

        Args:
            data_name (:obj:`str`): Name of the data element on which coordinate optimization is performed.
            objective (:obj:`str` or :obj:`callable`):
            processes (:obj:`int`): Number of parallel processes.
            **kwargs: keyword arguments which are passed to `Optimizer.optimize`
        """

        kwargs = {'data_name': data_name, 'objective': objective, **kwargs}
        pool = mp.Pool(processes=processes)

        f = partial(worker, **kwargs)

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

    def execute(self, worker):
        """Apply worker function `worker` to all cell objects and returns the results"""
        res = map(worker, self)

        return res

    def execute_mp(self, worker, processes=None):
        """Applies the worker function `worker` to all cells objects and returns the results using parallel computing."""
        pool = mp.Pool(processes=processes)
        res = pool.map(worker, self)

        pool.close()
        pool.join()

        return res

    def append(self, cell_obj):
        """Append Cell object `cell_obj` to the list of cells."""
        assert isinstance(cell_obj, Cell)
        self.cell_list = np.append(self.cell_list, cell_obj)

    def r_dist(self, stop, step, data_name='', norm_x=False, storm_weight=False, xlim=None):
        """ Calculates the radial distribution of a given data element for all cells in the `CellList`.

        Args:
            stop (:obj:`float`): Until how far from the cell spine the radial distribution should be calculated
            step (:obj:`float`): The binsize of the returned radial distribution
            data_name (:obj:`str`): The name of the data element on which to calculate the radial distribution
            norm_x (:obj:`bool`): If `True` the returned distribution will be normalized with the cell's radius set to 1.
            xlim (:obj:`str`): If `None`, all datapoints are taking into account. This can be limited by providing the
                value `full` (omit poles only), 'poles' (include only poles), or a float value which will limit the data
                points with around the midline where xmid - xlim < x < xmid + xlim.
            storm_weight (obj:`bool`): Only applicable for analyzing STORM-type data elements. If `True` the returned
                histogram is weighted with the number of photons measured.

        Returns:
            (tuple): tuple containing:

                xvals (:class:`~numpy.ndarray`) Array of distances from the cell midline, values are the middle of the bins
                out_arr (:class:`~numpy.ndarray`) Matrix of in bin heights
        """
        #todo might be a good idea to warm the user when attempting this on a  list of 3D data
        numpoints = len(np.arange(0, stop+step, step))
        out_arr = np.zeros((len(self), numpoints))
        for i, c in enumerate(self):
            xvals, yvals = c.r_dist(stop, step, data_name=data_name, norm_x=norm_x, storm_weight=storm_weight, xlim=xlim)
            out_arr[i] = yvals

        return xvals, out_arr

    def l_dist(self, nbins, data_name='', norm_x=False, r_max=None, storm_weight=False):
        y_arr = np.zeros((len(self), nbins))
        x_arr = np.zeros((len(self), nbins))
        for i, c in enumerate(self):
            xvals, yvals = c.l_dist(nbins, data_name=data_name, norm_x=norm_x, r_max=r_max, storm_weight=storm_weight)
            x_arr[i] = xvals
            y_arr[i] = yvals

        return x_arr, y_arr

    def l_classify(self, data_name=''):
        """Classifies foci in STORM-type data by they x-position along the long axis.

        The spots are classfied into 3 categories: 'poles', 'between' and 'mid'. The pole category are spots who are to
            the left and right of xl and xr, respectively. The class 'mid' is a section in the middle of the cell with a
            total length of half the cell's length, the class 'between' is the remaining two quarters between 'mid' and
            'poles'

        Args:
            data_name (:obj:`str`): Name of the STORM-type data element to classify. When its not specified the first
                STORM data element is used.

        Returns:
            :class:`~numpy.ndarray`: Array with number of spots in poles, between and mid classes, respectively, for
                each cell (rows)
        """

        return np.array([c.l_classify(data_name=data_name) for c in self])

    def a_dist(self):
        raise NotImplementedError()

    def get_intensity(self, mask='binary', data_name=''):
        """ Returns for all cells the mean fluorescence intensity in the region masked by either the binary image or synthetic
            binary image derived from the cell's coordinate system

        Args:
            mask (:obj:`str`): Either 'binary' or 'coords' to specify the source of the mask used.
                'binary' uses the binary imagea as mask, 'coords' uses reconstructed binary from coordinate system
            data_name (:obj:`str`): The name of the image data element to get the intensity values from.

        Returns:
            :class:`~numpy.ndarray`: Array of mean fluorescence pixel values

        """
        return np.array([c.get_intensity(mask=mask, data_name=data_name) for c in self])

    def measure_r(self, data_name='brightfield', in_place=True):
        """
        Measure the radius of the cell by finding the intensity-midpoint of the radial distribution derived from
        brightfield (default) or another data element.

        Args:
            data_name (:obj:`str`): Name of the data element to use.
            in_place (:obj:`bool`): If `True` the found value of `r` is directly substituted in the cell's coordinate
                system, otherwise the value is returned.

        Returns:
            :class:`~numpy.ndarray`: The measured radius `r` values if `in_place` is `False`, otherwise `None`.
        """

        r = [c.measure_r(data_name=data_name, in_place=in_place) for c in self]
        if not in_place:
            return np.array(r)

    def copy(self):
        """
        Make a copy of the `CellList` object and all its associated data elements.

        Returns:
            :class:`CellList`: Copied `CellList` object
        """
        return CellList([cell.copy() for cell in self])

    @property
    def radius(self):
        """:class:`~numpy.ndarray` Array of cell's radii in pixels"""
        return np.array([c.radius for c in self])

    @property
    def length(self):
        """:class:`~numpy.ndarray` Array of cell's lengths in pixels"""
        return np.array([c.length for c in self])

    @property
    def circumference(self):
        """:class:`~numpy.ndarray`: Array of cell's circumference in pixels"""
        return np.array([c.circumference for c in self])

    @property
    def area(self):
        """:class:`~numpy.ndarray`: Array of cell's area in square pixels"""
        return np.array([c.area for c in self])

    @property
    def surface(self):
        """:class:`~numpy.ndarray`: Array of cell's surface area (3d) in square pixels"""
        return np.array([c.surface for c in self])

    @property
    def volume(self):
        """:class:`~numpy.ndarray`: Array of cell's volume in cubic pixels"""
        return np.array([c.volume for c in self])

    @property
    def name(self):
        """:class:`~numpy.ndarray`: Array of cell's names"""
        return np.array([c.name for c in self])

    def __len__(self):
        return self.cell_list.__len__()

    def __iter__(self):
        return self.cell_list.__iter__()

    def __getitem__(self, key):
        if type(key) == int:
            return self.cell_list.__getitem__(key)
        else:
            return CellList(self.cell_list.__getitem__(key))

    def __setitem__(self, key, value):
        self.cell_list.__setitem__(key, value)

    def __delitem__(self, key):
        self.cell_list.__delitem__(key)

    def __contains__(self, item):
        return self.cell_list.__contains__(item)


def _solve_general(a, b, c, d):
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
    d0 = b ** 2. - 3. * a * c
    d1 = 2. * b ** 3. - 9. * a * b * c + 27. * a ** 2. * d

    r0 = np.square(d1) - 4. * d0 ** 3.
    r1 = (d1 + np.sqrt(r0)) / 2
    dc = np.cbrt(r1)  # power (1/3) gives nan's for coeffs [1.98537881e+01, 1.44894594e-02, 2.38096700e+00]01, 1.44894594e-02, 2.38096700e+00]
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
    try:
        assert(np.all(x_r > 0)) # dont know if this is guaranteed otherwise boundaries need to be passed and choosing from 3 slns
    except AssertionError:
        print(x_r)
        #todo find out of this is bad or not
        print('warning!')
        #raise ValueError
    return x_r


def _solve_len(x, xl, l, coeff):
    a0, a1, a2 = coeff

    l0 = (1 / (4 * a2)) * (
            ((a1 + 2 * a2 * x) * np.sqrt(1 + (a1 + 2 * a2 * x) ** 2) + np.arcsinh((a1 + 2 * a2 * x))) -
            ((a1 + 2 * a2 * xl) * np.sqrt(1 + (a1 + 2 * a2 * xl) ** 2) + np.arcsinh((a1 + 2 * a2 * xl)))
    )

    return l0 - l


def _calc_len(xl, xr, coeff):
    a0, a1, a2 = coeff
    l = (1 / (4 * a2)) * (
            ((a1 + 2 * a2 * xr) * np.sqrt(1 + (a1 + 2 * a2 * xr) ** 2) + np.arcsinh((a1 + 2 * a2 * xr))) -
            ((a1 + 2 * a2 * xl) * np.sqrt(1 + (a1 + 2 * a2 * xl) ** 2) + np.arcsinh((a1 + 2 * a2 * xl)))
    )

    return l