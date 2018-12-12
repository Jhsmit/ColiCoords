from colicoords.fitting import CellFit
from colicoords.support import allow_scalars, box_mean, running_mean
from colicoords.minimizers import Powell
import numbers
import mahotas as mh
import numpy as np
import operator
from functools import partial
from scipy.integrate import quad
from scipy.optimize import brentq
import multiprocess as mp
from tqdm.auto import tqdm


class Cell(object):
    """ColiCoords' main single-cell object.

    This class organizes all single-cell associated data together with an internal coordinate system.

    Parameters
    ----------
    data_object : :class:`~colicoords.data_models.Data`
        Holds all data describing this single cell.
    coords : :class:`Coordinates`
        Calculates transformations from/to cartesian and cellular coordinates.
    name : :obj:`str`
        Name identifying the cell (optional).
    **kwargs:
        Additional kwargs passed to :class:`~colicoords.cell.Coordinates`.

    Attributes
    ----------
    data : :class:`~colicoords.data_models.Data`
        Holds all data describing this single cell.
    coords : :class:`Coordinates`
        Calculates and optimizes the cell's coordinate system.
    name : :obj:`str`
        Name identifying the cell (optional).
    """
    def __init__(self, data_object, name=None, init_coords=True, **kwargs):
        self.data = data_object
        self.coords = Coordinates(self.data, initialize=init_coords, **kwargs)
        self.name = name

    def optimize(self, data_name='binary', cell_function=None, minimizer=Powell, **kwargs):
        """
        Optimize the cell's coordinate system.

        The optimization is performed on the data element given by ``data_name`` using the function `cell_function`.
        A default function depending on the data class is used of objective is omitted.

        Parameters
        ----------
        data_name : :obj:`str`, optional
            Name of the data element to perform optimization on.
        cell_function
            Optional subclass of :class:`~colicoords.fitting.CellMinimizeFunctionBase` to use as objective function.
        minimizer : Subclass of :class:`symfit.core.minimizers.BaseMinimizer` or :class:`~collections.abc.Sequence`
            Minimizer to use for the optimization. Default is the ``Powell`` minimizer.
        **kwargs :
            Additional kwargs are passed to :meth:`~colicoords.fitting.CellFit.execute`.

        Returns
        -------
        result : :class:`~symfit.core.fit_results.FitResults`
            ``symfit`` fit results object.


        """
        fit = CellFit(self, data_name=data_name, cell_function=cell_function, minimizer=minimizer)
        return fit.execute(**kwargs)

    @property
    def radius(self):
        """:obj:`float`: Radius of the cell in pixels."""
        return self.coords.r

    @property
    def length(self):
        """:obj:`float`: Length of the cell in pixels."""
        a0, a1, a2 = self.coords.coeff
        xl, xr = self.coords.xl, self.coords.xr
        l = (1 / (4 * a2)) * (
                ((a1 + 2 * a2 * xr) * np.sqrt(1 + (a1 + 2 * a2 * xr) ** 2) + np.arcsinh((a1 + 2 * a2 * xr))) -
                ((a1 + 2 * a2 * xl) * np.sqrt(1 + (a1 + 2 * a2 * xl) ** 2) + np.arcsinh((a1 + 2 * a2 * xl)))
        )

        return l

    @property
    def circumference(self):
        """:obj:`float`: Circumference of the cell in pixels."""

        # http://tutorial.math.lamar.edu/Classes/CalcII/ParaArcLength.aspx
        def integrant_top(t, a1, a2, r):
            return np.sqrt(1 + (a1 + 2 * a2 * t) ** 2 + ((4 * a2 ** 2 * r ** 2) / (1 + (a1 + 2 * a2 * t) ** 2) ** 2) + (
                        (4 * a2 * r) / np.sqrt(1 + (a1 + 2 * a2 * t))))

        def integrant_bot(t, a1, a2, r):
            return np.sqrt(1 + (a1 + 2 * a2 * t) ** 2 + ((4 * a2 ** 2 * r ** 2) / (1 + (a1 + 2 * a2 * t) ** 2) ** 2) - (
                        (4 * a2 * r) / np.sqrt(1 + (a1 + 2 * a2 * t))))

        top, terr = quad(integrant_top, self.coords.xl, self.coords.xr,
                         args=(self.coords.a1, self.coords.a2, self.coords.r))
        bot, berr = quad(integrant_bot, self.coords.xl, self.coords.xr,
                         args=(self.coords.a1, self.coords.a2, self.coords.r))

        return top + bot + 2 * np.pi * self.coords.r

    @property
    def area(self):
        """:obj:`float`: Area (2d) of the cell in square pixels."""
        return 2 * self.length * self.coords.r + np.pi * self.coords.r ** 2

    @property
    def surface(self):
        """:obj:`float`: Total surface area (3d) of the cell in square pixels."""
        return self.length * 2 * np.pi * self.coords.r + 4 * np.pi * self.coords.r ** 2

    @property
    def volume(self):
        """:obj:`float`: Volume of the cell in cubic pixels."""
        return np.pi * self.coords.r ** 2 * self.length + (4 / 3) * np.pi * self.coords.r ** 3

    def a_dist(self):
        raise NotImplementedError()

    def l_dist(self, nbins, start=None, stop=None, data_name='', norm_x=False, l_mean=None, r_max=None, storm_weight=False,
               method='gauss', sigma=0.5):
        """
        Calculates the longitudinal distribution of signal for a given data element.

        Parameters
        ----------
        nbins : :obj:`int`
            Number of bins between `start` and `stop`.
        start : :obj:`float`
            Distance from `xl` as starting point for the distribution, units are either pixels or normalized units
            if `norm_x=True`.
        stop : :obj:`float`
            Distance from `xr` as end point for the distribution, units are are either pixels or normalized units
            if `norm_x=True`.
        bins : :class:`~numpy.ndarray`
            Array of bin edges to use. Overrrides `nbin`, `start` and `stop`.
        data_name : :obj:`str`
            Name of the data element to use.
        norm_x : :obj:`bool`
            If `True` the output distribution will be normalized.
        l_mean : :obj:`float`, optional
            When `norm_x` is `True`, all length coordinates are divided by the length of the cell to normalize it. If
            `l_mean` is given, the length coordinates at the poles are divided by `l_mean` instead to allow equal scaling
            of all pole regions.
        r_max : :obj:`float`, optional
            Datapoints within r_max from the cell midline will be included. If `None` the value from the cell's
            coordinate system will be used.
        storm_weight : :obj:`bool`
            If `True` the datapoints of the specified STORM-type data will be weighted by their intensity.
        method : :obj:`str`
            Method of averaging datapoints to calculate the final distribution curve.
        sigma : :obj:`float`
            Applies only when `method` is set to 'gauss'. `sigma` gives the width of the gaussian used for convoluting
            datapoints.

        Returns
        -------
        xvals : :class:`~numpy.ndarray`
            Array of distances along the cell midline, values are the middle of the bins/kernel.
        yvals : :class:`~numpy.ndarray`
            Array of bin heights.

        """
        length = 1 if norm_x else self.length
        r_max = r_max if r_max else self.coords.r
        stop = 1.25 * length if not stop else stop
        start = -0.25 * length if not start else start  # also needs to be uniform with l_mean? no

        if not data_name:
            try:
                data_elem = list(self.data.flu_dict.values())[0]  # yuck
            except IndexError:
                try:
                    data_elem = list(self.data.storm_dict.values())[0]
                except IndexError:
                    raise IndexError('No valid data element found')
        else:
            try:
                data_elem = self.data.data_dict[data_name]
            except KeyError:
                raise ValueError('Chosen data not found')

        bins = np.linspace(start, stop, num=nbins, endpoint=True)

        if data_elem.ndim == 1:
            assert data_elem.dclass == 'storm'
            xp = data_elem['x']
            yp = data_elem['y']

            idx_left, idx_right, xc = self.coords.get_idx_xc(xp, yp)

        elif data_elem.ndim == 2 or data_elem.ndim == 3:  # image data
            xp, yp = self.coords.x_coords, self.coords.y_coords
            idx_left, idx_right, xc = self.coords.get_idx_xc(xp, yp)

        else:
            raise ValueError('Invalid data element dimensions')

        r = self.coords.calc_rc(xp, yp)
        bools = r < r_max

        # todo update to calc_lc
        x_len = calc_lc(self.coords.xl, xc[bools].flatten(), self.coords.coeff)
        if norm_x:
            if l_mean:

                len_norm = x_len / self.length
                len_norm[x_len < 0] = x_len[x_len < 0] / l_mean
                len_norm[x_len > self.length] = ((x_len[x_len > self.length] - self.length) / l_mean) + 1

                x_len = len_norm
            else:
                x_len = x_len / self.length

        if method == 'gauss' and data_elem.dclass == 'storm':
            print("Warning: method 'gauss' is not a storm-compatible method, method was set to 'box'")
            method = 'box'

        if method == 'gauss':
            bin_func = running_mean
            bin_kwargs = {'sigma': sigma}
            xvals = bins
        elif method == 'box':
            bools = (x_len > bins.min()) * (x_len < bins.max())  # Remove values outside of bins range
            x_len = x_len[bools]

            bin_func = box_mean
            bin_kwargs = {'storm_weight': storm_weight}
            xvals = bins + 0.5 * np.diff(bins)[0]
        else:
            raise ValueError('Invalid method')

        if data_elem.ndim == 1:
            y_weight = data_elem['intensity'][bools] if storm_weight else None
            yvals = bin_func(x_len, y_weight, bins, **bin_kwargs)

        elif data_elem.ndim == 2:
            y_weight = np.clip(data_elem[bools].flatten(), 0, None)  # Negative values are set to zero
            yvals = bin_func(x_len, y_weight, bins, **bin_kwargs)

        elif data_elem.ndim == 3:
            yvals = np.array([bin_func(x_len, y_weight[bools].flatten(), bins, **bin_kwargs) for y_weight in data_elem])

        return xvals, yvals

    def l_classify(self, data_name=''):
        """
        Classifies foci in STORM-type data by they x-position along the long axis.

        The spots are classified into 3 categories: 'poles', 'between' and 'mid'. The pole category are spots who are to
        the left and right of xl and xr, respectively. The class 'mid' is a section in the middle of the cell with a
        total length of half the cell's length, the class 'between' is the remaining two quarters between 'mid' and
        'poles'.

        Parameters
        ----------
        data_name : :obj:`str`
            Name of the STORM-type data element to classify. When its not specified the first STORM data element is used.

        Returns
        -------
        l_classes : :obj:`tuple`
            Tuple with number of spots in poles, between and mid classes, respectively.
        """

        if not data_name:
            data_elem = list(self.data.storm_dict.values())[0]
        else:
            data_elem = self.data.data_dict[data_name]
            assert data_elem.dclass == 'storm'

        x, y = data_elem['x'], data_elem['y']
        lc = self.coords.calc_lc(x, y)
        lq1 = self.length / 4
        lq3 = 3 * lq1

        poles = np.sum(lc <= 0) + np.sum(lc >= self.length)
        between = np.sum(np.logical_and(lc > 0, lc < lq1)) + np.sum(np.logical_and(lc < self.length, lc > lq3))
        mid = np.sum(np.logical_and(lc >= lq1, lc <= lq3))

        try:
            assert len(x) == (poles + between + mid)
        except AssertionError:
            raise ValueError("Invalid number of points")

        return poles, between, mid

    def r_dist(self, stop, step, data_name='', norm_x=False, limit_l=None, storm_weight=False, method='gauss',
               sigma=0.3):
        """
        Calculates the radial distribution of a given data element.

        Parameters
        ----------
        stop : :obj:`float`
            Until how far from the cell spine the radial distribution should be calculated.
        step : :obj:`float`
            The binsize of the returned radial distribution.
        data_name : :obj:`str`
            The name of the data element on which to calculate the radial distribution.
        norm_x : :obj:`bool`
            If `True` the returned distribution will be normalized with the cell's radius set to 1.
        limit_l : :obj:`str`
            If `None`, all datapoints are used. This can be limited by providing the value `full` (omit poles only),
            'poles' (include only poles), or a float value between 0 and 1 which will limit the data points by
            longitudinal coordinate around the midpoint of the cell.
        storm_weight : :obj:`bool`
            Only applicable for analyzing STORM-type data elements. If `True` the returned histogram is weighted with
            the values in the 'Intensity' field.
        method : :obj:`str`, either 'gauss' or 'box'
            Method of averaging datapoints to calculate the final distribution curve.
        sigma : :obj:`float`
            Applies only when `method` is set to 'gauss'. `sigma` gives the width of the gaussian used for convoluting
            datapoints.

        Returns
        -------
        xvals : :class:`~numpy.ndarray`
            Array of distances from the cell midline, values are the middle of the bins.
        yvals : :class:`~numpy.ndarray`
            Array of in bin heights.
        """

        if not data_name:
            try:
                data_elem = list(self.data.flu_dict.values())[0]  # yuck
            except IndexError:
                try:
                    data_elem = list(self.data.storm_dict.values())[0]
                except IndexError:
                    raise IndexError('No valid data element found')
        else:
            try:
                data_elem = self.data.data_dict[data_name]
            except KeyError:
                raise ValueError('Chosen data not found')

        if method == 'gauss' and data_elem.dclass == 'storm':
            print("Warning: method 'gauss' is not a storm-compatible method, method was set to 'box'")
            method = 'box'

        bins = np.arange(0, stop + step, step)

        if method == 'gauss':
            bin_func = running_mean
            bin_kwargs = {'sigma': sigma}
            xvals = bins
        elif method == 'box':
            bin_func = box_mean
            bin_kwargs = {'storm_weight': storm_weight}
            bins = np.arange(0, stop + step, step)
            xvals = bins + 0.5 * step  # xval is the middle of the bin
        else:
            raise ValueError('Invalid method')

        if data_elem.ndim == 1:
            assert data_elem.dclass == 'storm'
            x = data_elem['x']
            y = data_elem['y']
            xc = self.coords.calc_xc(x, y)

            r = self.coords.calc_rc(x, y)
            r = r / self.coords.r if norm_x else r
            y_weight = data_elem['intensity'] if storm_weight else None

        elif data_elem.ndim == 2 or data_elem.ndim == 3:
            r = (self.coords.rc / self.coords.r if norm_x else self.coords.rc)
            xc = self.coords.xc
            x = self.coords.x_coords
            y = self.coords.y_coords
            y_weight = data_elem

        else:
            raise ValueError("Invalid data dimensions")

        if limit_l:
            if limit_l == 'full':
                b = (xc > self.coords.xl) * (xc < self.coords.xr).astype(bool)
            elif limit_l == 'poles':
                b = ((xc <= self.coords.xl) * (xc >= self.coords.xr)).astype(bool)
            else:
                assert 0 < limit_l < 1
                mid_l = self.length / 2
                lc = self.coords.calc_lc(x, y)
                limit = limit_l * self.length

                b = ((lc > mid_l - limit / 2) * (lc < mid_l + limit / 2)).astype(bool)
        else:
            b = True

        if data_elem.ndim <= 2:
            y_wt = y_weight[b].flatten() if y_weight is not None else None
            yvals = bin_func(r[b].flatten(), y_wt, bins, **bin_kwargs)
        else:
            yvals = np.vstack([bin_func(r[b].flatten(), d[b].flatten(), bins) for d in data_elem])

        return xvals, yvals

    def measure_r(self, data_name='brightfield', mode='max', in_place=True, **kwargs):
        """
        Measure the radius of the cell.

        The radius is found by the intensity-mid/min/max-point of the radial distribution derived from brightfield
        (default) or another data element.

        Parameters
        ----------
        data_name : :obj:`str`
            Name of the data element to use.
        mode : :obj:`str`
            Mode to find the radius. Can be either 'min', 'mid' or 'max' to use the minimum, middle or maximum value
            of the radial distribution, respectively.
        in_place : :obj:`bool`
            If `True` the found value of `r` is directly substituted in the cell's coordinate system, otherwise the
            value is returned.

        Returns
        -------
        radius : :obj:`float`
            The measured radius `r` if `in_place` is `False`, otherwise `None`.
        """

        step = kwargs.pop('step', 1)
        stop = kwargs.pop('stop', int(self.data.shape[0] / 2))
        x, y = self.r_dist(stop, step, data_name=data_name)  # todo again need sensible default for stop

        if mode == 'min':
            imin = np.argmin(y)
            r = x[imin]
        elif mode == 'mid':
            mid_val = (np.min(y) + np.max(y)) / 2
            imin = np.argmin(y)
            imax = np.argmax(y)
            y_select = y[imin:imax] if imax > imin else y[imax:imin][::-1]
            x_select = x[imin:imax] if imax > imin else x[imax:imin][::-1]

            try:
                assert np.all(np.diff(y_select) > 0)
            except AssertionError:
                print('Radial distribution not monotonically increasing')
            try:
                r = np.interp(mid_val, y_select, x_select)
            except ValueError:
                print("r value not found")
                return
        elif mode == 'max':
            imax = np.argmax(y)
            r = x[imax]
        else:
            ValueError('Invalid value for mode')

        if in_place:
            self.coords.r = r
        else:
            return r

    def reconstruct_image(self, data_name, norm_x=False, r_scale=1, **kwargs):
        # todo stop and step defaults when norm_x=True?
        # todo allow reconstruction of standardized cell shape
        # todo refactor to reconstruct image?
        """
        Reconstruct the image from a given data element and the cell's current coordinate system.

        Parameters
        ----------
        data_name : :obj:`str`
            Name of the data element to use.
        norm_x : :obj:`bool`
            Boolean indicating whether or not to normalize to r=1.
        r_scale : :obj:`float`
            Stretch or compress the image in the radial direction by this factor.
        **kwargs
            Optional keyword arguments are 'stop' and 'step' which are passed to `r_dist`.

        Returns
        -------
        img : :class:`~numpy.ndarray`
            Image of the reconstructed cell.
        """

        stop = kwargs.pop('stop', np.ceil(np.max(self.data.shape) / 2))
        step = kwargs.pop('step', 1)

        xp, fp = self.r_dist(stop, step, data_name=data_name, norm_x=norm_x)
        interp = np.interp(r_scale * self.coords.rc, xp, np.nan_to_num(fp))  # todo check nantonum cruciality

        return interp

    def get_intensity(self, mask='binary', data_name='', func=np.mean):
        """
        Returns the mean fluorescence intensity.

        Mean fluorescence intensity either in the region masked by the binary image or reconstructed binary image derived
        from the cell's coordinate system.

        Parameters
        ----------
        mask : :obj:`str`
            Either 'binary' or 'coords' to specify the source of the mask used. 'binary' uses the binary image as mask,
            'coords' uses reconstructed binary from coordinate system.
        data_name : :obj:`str`:
            The name of the image data element to get the intensity values from.
        func : :obj:`callable`
            This function is applied to the data elements pixels selected by the masking operation. The default is
            `np.mean()`.

        Returns
        -------
        value : :obj:`float`:
            Mean fluorescence pixel value.
        """

        if mask == 'binary':
            m = self.data.binary_img.astype(bool)
        elif mask == 'coords':
            m = self.coords.rc < self.coords.r
        else:
            raise ValueError("Mask keyword should be either 'binary' or 'coords'")

        if not data_name:
            data_elem = list(self.data.flu_dict.values())[0]  # yuck
        else:
            try:
                data_elem = self.data.data_dict[data_name]
            except KeyError:
                raise ValueError('Chosen data not found')

        return func(data_elem[m])

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
        Make a copy of the cell object and all its associated data elements.

        This is a deep copy meaning that all numpy data arrays are copied in memory and therefore modifying the copied
        cell object does not modify the original cell object.

        Returns
        -------
        cell : :class:`~colicoords.cell.Cell`:
            Copied cell object.

        """
        # todo needs testing (this is done?) arent there more properties to copy?
        parameters = {par: getattr(self.coords, par) for par in self.coords.parameters}
        parameters['shape'] = self.coords.shape
        new_cell = Cell(data_object=self.data.copy(), name=self.name, init_coords=False, **parameters)

        return new_cell


class Coordinates(object):
    """
    Cell's coordinate system described by the polynomial p(x) and associated functions.

    Parameters
    ----------
    data : :class:`~colicoords.data_models.Data`
        The `data` object defining the shape.
    initialize : :obj:`bool`, optional
        If `False` the coordinate system parameters are not initialized with initial guesses.
    **kwargs
        Can be used to manually supply parameter values if `initialize` is `False`.

    Attributes
    ----------
    xl : :obj:`float`
        Left cell pole x-coordinate.
    xr : :obj:`float`
        Right cell pole x-coordinate.
    r : :obj:`float`
        Cell radius.
    coeff : :class:`~numpy.ndarray`
        Coefficients [a0, a1, a2] of the polynomial a0 + a1*x + a2*x**2 which describes the cell's shape.
    """

    parameters = ['r', 'xl', 'xr', 'a0', 'a1', 'a2']

    def __init__(self, data, initialize=True, **kwargs):
        self.data = data
        self.coeff = np.array([1., 1., 1.])

        if initialize:
            self.xl, self.xr, self.r, self.coeff = self._initial_guesses(data)  # refactor to class method
            self.coeff = self._initial_fit()
            self.shape = data.shape
        else:
            for p in self.parameters + ['shape']:
                setattr(self, p, kwargs.pop(p, 1))

    @property
    def a0(self):
        """float: Polynomial p(x) 0th degree coefficient."""
        return self.coeff[0]

    @a0.setter
    def a0(self, value):
        self.coeff[0] = value

    @property
    def a1(self):
        """float: Polynomial p(x) 1st degree coefficient."""
        return self.coeff[1]

    @a1.setter
    def a1(self, value):
        self.coeff[1] = value

    @property
    def a2(self):
        """float: Polynomial p(x) 2nd degree coefficient."""
        return self.coeff[2]

    @a2.setter
    def a2(self, value):
        self.coeff[2] = value

    def sub_par(self, par_dict):
        """
        Substitute the values in `par_dict` as the coordinate systems parameters.

        Parameters
        ----------
        par_dict : :obj:`dict`
            Dictionary with parameters which values are set to the attributes.
        """
        for k, v in par_dict.items():
            setattr(self, k, v)

    @allow_scalars
    def calc_xc(self, xp, yp):
        """
        Calculates the coordinate xc on p(x) closest to xp, yp.
        
        All coordinates are cartesian. Solutions are found by solving the cubic equation.

        Parameters
        ----------
        xp : :obj:`float` or :class:`~numpy.ndarray`
            Input scalar or vector/matrix x-coordinate. Must be the same shape as yp.
        yp : :obj:`float` : or :class:`~numpy.ndarray`
            Input scalar or vector/matrix x-coordinate. Must be the same shape as xp.

        Returns
        -------
        xc : :obj:`float` or :class:`~numpy.ndarray`
            Cellular x-coordinate for point(s) xp, yp
        """

        assert xp.shape == yp.shape
        # https://en.wikipedia.org/wiki/Cubic_function#Algebraic_solution
        a0, a1, a2 = self.coeff
        # xp, yp = xp.astype('float32'), yp.astype('float32')
        # Converting of cell spine polynomial coefficients to coefficients of polynomial giving distance r
        a, b, c, d = 4 * a2 ** 2, 6 * a1 * a2, 4 * a0 * a2 + 2 * a1 ** 2 - 4 * a2 * yp + 2, 2 * a0 * a1 - 2 * a1 * yp - 2 * xp
        # a: float, b: float, c: array, d: array
        discr = 18 * a * b * c * d - 4 * b ** 3 * d + b ** 2 * c ** 2 - 4 * a * c ** 3 - 27 * a ** 2 * d ** 2

        # if np.any(discr == 0):
        #     raise ValueError('Discriminant equal to zero encountered. This should never happen. Please make an issue.')

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

    @allow_scalars
    def calc_xc_mask(self, xp, yp):
        """
        Calculated whether point (xp, yp) is in either the left or right polar areas, or in between.

        Returned values are 1 for left pole, 2 for middle, 3 for right pole.

        Parameters
        ----------
        xp : :obj:`float` or :class:`~numpy.ndarray`
            Input scalar or vector/matrix x-coordinate. Must be the same shape as yp.
        yp : :obj:`float` or :class:`~numpy.ndarray`
            Input scalar or vector/matrix x-coordinate. Must be the same shape as xp.

        Returns
        -------
        xc_mask : :obj:`float`: or :class:`~numpy.ndarray`:
            Array to mask different cellular regions.
        """

        idx_left, idx_right, xc = self.get_idx_xc(xp, yp)
        mask = 2 * np.ones_like(xp)
        mask[idx_left] = 1
        mask[idx_right] = 3

        return mask

    @allow_scalars
    def calc_xc_masked(self, xp, yp):
        """
        Calculates the coordinate xc on p(x) closest to (xp, yp), where xl < xc < xr.

        Parameters
        ----------
        xp : :obj:`float`: or :class:`~numpy.ndarray`:
            Input scalar or vector/matrix x-coordinate. Must be the same shape as yp.
        yp : :obj:`float`: or :class:`~numpy.ndarray`:
            Input scalar or vector/matrix x-coordinate. Must be the same shape as xp.

        Returns
        -------
        xc_mask : :obj:`float` or :class:`~numpy.ndarray`
            Cellular x-coordinate for point(s) xp, yp, where xl < xc < xr.
        """
        idx_left, idx_right, xc = self.get_idx_xc(xp, yp)
        xc[idx_left] = self.xl
        xc[idx_right] = self.xr

        return xc

    @allow_scalars
    def calc_rc(self, xp, yp):
        """
        Calculates the distance of (xp, yp) to (xc, p(xc)).

        The returned value is the distance from the points (xp, yp) to the midline of the cell.

        Parameters
        ----------
        xp : :obj:`float` or :class:`~numpy.ndarray`
            Input scalar or vector/matrix x-coordinate. Must be the same shape as yp.
        yp : :obj:`float` or :class:`~numpy.ndarray`
            Input scalar or vector/matrix x-coordinate. Must be the same shape as xp.

        Returns
        -------
        rc : :obj:`float` or :class:`~numpy.ndarray`
            Distance to the midline of the cell.
        """

        xc = self.calc_xc_masked(xp, yp)
        a0, a1, a2 = self.coeff
        return np.sqrt((xc - xp) ** 2 + (a0 + xc * (a1 + a2 * xc) - yp) ** 2)

    @allow_scalars
    def calc_lc(self, xp, yp):
        """
        Calculates distance of xc along the midline the cell corresponding to the points (xp, yp).

        The returned value is the distance from the points (xp, yp) to the midline of the cell.

        Parameters
        ----------
        xp : :obj:`float`: or :class:`~numpy.ndarray`
            Input scalar or vector/matrix x-coordinate. Must be the same shape as yp.
        yp : :obj:`float` or :class:`~numpy.ndarray`
            Input scalar or vector/matrix x-coordinate. Must be the same shape as xp.

        Returns
        -------
        lc : :obj:`float` or :class:`~numpy.ndarray`
            Distance along the midline of the cell.
        """

        xc = self.calc_xc_masked(xp, yp)
        return calc_lc(self.xl, xc, self.coeff)

    @allow_scalars
    def calc_phi(self, xp, yp):
        """
        Calculates the angle between the line perpendical to the cell midline and the line between (xp, yp)
        and (xc, p(xc)).

        The returned values are in degrees. The angle is defined to be 0 degrees for values in the upper half of the
        image (yp < p(xp)), running from 180 to zero along the right polar region, 180 degrees in the lower half and
        running back to 0 degrees along the left polar region.

        Parameters
        ----------
        xp : :obj:`float` or :class:`~numpy.ndarray`
            Input scalar or vector/matrix x-coordinate. Must be the same shape as yp.
        yp : :obj:`float` or :class:`~numpy.ndarray`
            Input scalar or vector/matrix x-coordinate. Must be the same shape as xp.

        Returns
        -------
        phi : :obj:`float` or :class:`~numpy.ndarray`
            Angle phi for (xp, yp).
        """
        idx_left, idx_right, xc = self.get_idx_xc(xp, yp)
        xc[idx_left] = self.xl
        xc[idx_right] = self.xr
        yc = self.p(xc)

        phi = np.empty(xp.shape)
        top = yp < self.p(xp)
        phi[top] = 0
        phi[~top] = np.pi

        th1 = np.arctan2(yp - yc, xc - xp)
        th2 = np.arctan(self.p_dx(xc))
        thetha = th1 + th2 + np.pi / 2
        phi[idx_right] = (np.pi - thetha[idx_right]) % np.pi
        phi[idx_left] = thetha[idx_left]

        return phi * (180 / np.pi)

    def get_idx_xc(self, xp, yp):
        """
        Finds the indices of the arrays xp an yp where they either belong to the left or right polar regions,
        as well as coordinates xc.

        Parameters
        ----------
        xp : :class:`~numpy.ndarray`
            Input  x-coordinates. Must be the same shape as yp.
        yp : :class:`~numpy.ndarray`
            Input y-coordinates. Must be the same shape as xp.

        Returns
        -------
        idx_left : :class:`~numpy.ndarray`
            Index array of elements in the area of the cell's left pole.
        idx_right : :class:`~numpy.ndarray`
            Index array of elements in the area of cell's right pole.
        xc : :class:`~numpy.ndarray`
            Cellular coordinates `xc` corresponding to `xp`, `yp`, extending into the polar regions.
        """

        xc = np.array(self.calc_xc(xp, yp).copy())
        yp = self.p(xc)

        # Area left of perpendicular line at xl:
        op = operator.lt if self.p_dx(self.xl) > 0 else operator.gt
        idx_left = op(yp, self.q(xc, self.xl))

        op = operator.gt if self.p_dx(self.xr) > 0 else operator.lt
        idx_right = op(yp, self.q(xc, self.xr))

        return idx_left, idx_right, xc

    @allow_scalars
    # todo scalar input wont work because of sqeeeeeze?
    def transform(self, xp, yp):
        """
        Transforms image coordinates (xp, yp) to cell coordinates (lc, rc, psi)

        Parameters
        ----------
        xp : :obj:`float` or :class:`~numpy.ndarray`
            Input scalar or vector/matrix x-coordinate. Must be the same shape as yp
        yp : :obj:`float` or :class:`~numpy.ndarray`
            Input scalar or vector/matrix x-coordinate. Must be the same shape as xp

        Returns
        -------
        coordinates : :obj:`tuple`
            Tuple of cellular coordinates lc, rc, psi
        """

        lc = self.calc_lc(xp, yp)
        rc = self.calc_rc(xp, yp)
        psi = self.calc_phi(xp, yp)

        return lc, rc, psi

    @allow_scalars
    def full_transform(self, xp, yp):
        """
        Transforms image coordinates (xp, yp) to cell coordinates (xc, lc, rc, psi).

        Parameters
        ----------
        xp : :obj:`float` or :class:`~numpy.ndarray`
            Input scalar or vector/matrix x-coordinate. Must be the same shape as yp.
        yp : :obj:`float` or :class:`~numpy.ndarray`
            Input scalar or vector/matrix x-coordinate. Must be the same shape as xp.

        Returns
        -------
        coordinates : :obj:`tuple`
            Tuple of cellular coordinates xc, lc, rc, psi.
        """

        xc = self.calc_xc_masked(xp, yp)
        lc = self.calc_lc(xp, yp)
        rc = self.calc_rc(xp, yp)
        psi = self.calc_phi(xp, yp)

        return xc, lc, rc, psi

    @allow_scalars
    def rev_transform(self, lc, rc, phi, l_norm=True):
        """
        Reverse transform from cellular coordinates `lc`, `rc`, `phi` to cartesian coordinates `xp`, `yp`.

        Parameters
        ----------
        lc : :obj:`float` or :class:`~numpy.ndarray`
             Input scalar or vector/matrix l-coordinate.
        rc : :obj:`float` or :class:`~numpy.ndarray`
            Input scalar or vector/matrix l-coordinate.
        phi : :obj:`float` or :class:`~numpy.ndarray`
            Input scalar or vector/matrix l-coordinate.
        l_norm : :obj:`bool`, optional
            If `True` (default), the lc coordinate has to be input as normalized.

        Returns
        -------
        xp : :obj:`float` or :class:`~numpy.ndarray`
            Cartesian x-coordinate corresponding to `lc`, `rc`, `phi`
        yp : :obj:`float` or :class:`~numpy.ndarray`
            Cartesian y-coordinate corresponding to `lc`, `rc`, `phi`
        """

        assert lc.min() >= 0
        if l_norm:
            assert lc.max() <= 1
            lc *= self.length

        else:
            assert lc.max() <= self.length

        b_left = lc <= 0
        b_right = lc >= self.length
        b_mid = np.logical_and(~b_left, ~b_right)

        xp, yp = np.empty_like(lc, dtype=float), np.empty_like(lc, dtype=float)

        # left:
        xc = self.xl
        yc = self.p(xc)

        th2 = np.arctan(self.p_dx(xc)) * (180 / np.pi)
        theta = 180 - (-th2 + phi[b_left])

        dx = -rc[b_left] * np.sin(theta * (np.pi / 180))
        dy = rc[b_left] * np.cos(theta * (np.pi / 180))

        xp[b_left] = xc + dx
        yp[b_left] = yc + dy

        # middle:
        # brute force fsolve xc form lc
        sign = (phi[b_mid] / -90) + 1  # top or bottom of the cell
        # xc = np.array([fsolve(solve_length, l_guess, args=(self.xl, self.coeff, l_guess)).squeeze() for l_guess in lc[b_mid]])
        xc = np.array(
            [brentq(solve_length, self.xl, self.xr, args=(self.xl, self.coeff, l_guess)) for l_guess in lc[b_mid]])

        # lc_mid = lc[b_mid].copy()
        # xc = fsolve(solve_length, lc_mid, args=(self.xl, self.coeff, lc_mid)).squeeze()

        yc = self.p(xc)

        p_dx_sq = self.p_dx(xc) ** 2
        dy = (-rc[b_mid] / np.sqrt(1 + p_dx_sq)) * sign
        dx = (rc[b_mid] / np.sqrt(1 + (1 / p_dx_sq))) * sign * np.sign(self.p_dx(xc))

        xp[b_mid] = xc + dx
        yp[b_mid] = yc + dy

        # right
        xc = self.xr
        yc = self.p(xc)

        th2 = np.arctan(self.p_dx(self.xr)) * (180 / np.pi)
        theta = 180 - (th2 + phi[b_right])

        dx = rc[b_right] * np.sin(theta * (np.pi / 180))
        dy = rc[b_right] * np.cos(theta * (np.pi / 180))

        xp[b_right] = xc + dx
        yp[b_right] = yc + dy

        return xp, yp

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
    def phi(self):
        """:class:`~numpy.ndarray`: Matrix of shape m x n equal to cell with angle psi relative to the cell midline."""
        return self.calc_phi(self.x_coords, self.y_coords)

    @property
    def length(self):
        """:obj:`float`: Length of the cell in pixels."""
        a0, a1, a2 = self.coeff
        xl, xr = self.xl, self.xr
        l = (1 / (4 * a2)) * (
                ((a1 + 2 * a2 * xr) * np.sqrt(1 + (a1 + 2 * a2 * xr) ** 2) + np.arcsinh((a1 + 2 * a2 * xr))) -
                ((a1 + 2 * a2 * xl) * np.sqrt(1 + (a1 + 2 * a2 * xl) ** 2) + np.arcsinh((a1 + 2 * a2 * xl)))
        )

        return l

    def p(self, x_arr):
        """
        Calculate p(x).

        The function p(x) describes the midline of the cell.

        Parameters
        ----------
        x_arr : :class:`~numpy.ndarray`
            Input x values.

        Returns
        -------
        p : :class:`~numpy.ndarray`
            Evaluated polynomial p(x)
        """
        a0, a1, a2 = self.coeff
        return a0 + a1 * x_arr + a2 * x_arr ** 2

    def p_dx(self, x_arr):
        """
        Calculate the derivative p'(x) evaluated at x.

        Parameters
        ----------
        x_arr :class:`~numpy.ndarray`:
            Input x values.

        Returns
        -------
        p_dx : :class:`~numpy.ndarray`
            Evaluated function p'(x).
        """

        a0, a1, a2 = self.coeff
        return a1 + 2 * a2 * x_arr

    def q(self, x, xp):
        """array_like: Returns q(x) where q(x) is the line perpendicular to p(x) at xp"""
        return (-x / self.p_dx(xp)) + self.p(xp) + (xp / self.p_dx(xp))

    def get_core_points(self, xl=None, xr=None):
        """
        Returns the coordinates of the roughly estimated 'core' points of the cell.

        Used for determining the initial guesses for the coefficients of p(x).

        Parameters
        ----------
        xl : :obj:`float`, optional
            Starting point x of where to get the 'core' points.
        xr : :obj:`float`, optional
            End point x of where to get the 'core' points.

        Returns
        -------
        xvals : :class:`np.ndarray`
            Array of x coordinates of 'core' points.
        yvals : :class:`np.ndarray`
            Array of y coordinates of 'core' points.
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
            l = (area - np.pi * r ** 2) / (2 * r)
            y_cen, x_cen = mh.center_of_mass(data.binary_img)
            xl, xr = x_cen - l / 2, x_cen + l / 2
            coeff = np.array([y_cen, 0.01, 0.0001])

        else:
            raise ValueError("Binary image is required for initial guesses of cell coordinates")

        return xl, xr, r, coeff

    def _initial_fit(self):
        x, y = self.get_core_points()
        return np.polyfit(x, y, 2)[::-1]


def optimize_worker(cell, **kwargs):
    """
    Worker object for optimize multiprocessing.

    Parameters
    ----------
    cell : :class:`~colicoords.cell.Cell`
        Cell object to optimize.
    **kwargs
        Additional keyword arguments passed to :meth:`~colicoords.cell.Cell.optimize`

    Returns
    -------
    result : :class:`~symit.core.fit import FitResults
    """
    res = cell.optimize(**kwargs)
    return res


class CellList(object):
    """
    List equivalent of the :class:`~colicoords.cell.Cell` object.

    This Object holding a list of cell objects exposing several methods to either apply functions to all cells or to
    extract values from all cell objects. It supports iteration over Cell objects and Numpy-style array indexing.

    Parameters
    ----------
    cell_list : :obj:`list` or :class:`numpy.ndarray`
        List of array of :class:`~colicoords.cell.Cell` objects.

    Attributes
    ----------
    cell_list : :class:`~numpy.ndarray`
        Numpy array of `Cell` objects

    """

    def __init__(self, cell_list):
        self.cell_list = np.array(cell_list)

    def optimize(self, data_name='binary', cell_function=None, minimizer=Powell, **kwargs):
        """
        Optimize the cell's coordinate system.

        The optimization is performed on the data element given by ``data_name``
        using objective function `objective`. A default depending on the data class is used of objective is omitted.

        Parameters
        ----------
        data_name : :obj:`str`, optional
            Name of the data element to perform optimization on.
        cell_function
            Optional subclass of :class:`~colicoords.fitting.CellMinimizeFunctionBase` to use as objective function.
        minimizer : Subclass of :class:`symfit.core.minimizers.BaseMinimizer` or :class:`~collections.abc.Sequence`
            Minimizer to use for the optimization. Default is the ``Powell`` minimizer.
        **kwargs :
            Additional kwargs are passed to :meth:`~colicoords.fitting.CellFit.execute`.

        Returns
        -------
        res_list : :obj:`list` of :class:`~symfit.core.fit_results.FitResults`
            List of `symfit` ``FitResults`` object.
        """

        return [c.optimize(data_name=data_name, cell_function=cell_function, minimizer=minimizer, **kwargs) for c in tqdm(self)]

    def optimize_mp(self, data_name='binary', cell_function=None, minimizer=Powell, processes=None, **kwargs):
        """ Optimize all cell's coordinate systems using `optimize` through parallel computing.

        A call to this method must be  protected by if __name__ == '__main__' if its not executed in jupyter notebooks.

        Parameters
        ----------
        data_name : :obj:`str`, optional
            Name of the data element to perform optimization on.
        cell_function
            Optional subclass of :class:`~colicoords.fitting.CellMinimizeFunctionBase` to use as objective function.
        minimizer : Subclass of :class:`symfit.core.minimizers.BaseMinimizer` or :class:`~collections.abc.Sequence`
            Minimizer to use for the optimization. Default is the ``Powell`` minimizer.
        processes : :obj:`int`
            Number of parallel processes to spawn. Default is the number of logical processors on the host machine.
        **kwargs :
            Additional kwargs are passed to :meth:`~colicoords.fitting.CellFit.execute`.

        Returns
        -------
        res_list : :obj:`list` of :class:`~symfit.core.fit_results.FitResults`
            List of `symfit` ``FitResults`` object.
        """

        kwargs = {'data_name': data_name, 'cell_function': cell_function, 'minimizer': minimizer, **kwargs}
        pool = mp.Pool(processes=processes)

        f = partial(optimize_worker, **kwargs)

        res = list(tqdm(pool.imap(f, self), total=len(self)))

        for r, cell in zip(res, self):
            cell.coords.sub_par(r.params)

        return res

    def execute(self, worker):
        """
        Apply worker function `worker` to all cell objects and returns the results.

        Parameters
        ----------
        worker : :obj:`callable`
            Worker function to be executed on all cell objects.

        Returns
        -------
        res : :obj:`list`
            List of resuls returned from `worker`
        """
        res = map(worker, self)

        return res

    def execute_mp(self, worker, processes=None, **kwargs):
        """
        Apply worker function `worker` to all cell objects and returns the results.

        Parameters
        ----------
        worker : :obj:`callable`
            Worker function to be executed on all cell objects.
        processes : :obj:`int`
            Number of parallel processes to spawn. Default is the number of logical processors on the host machine.


        Returns
        -------
        res : :obj:`list`
            List of results returned from ``worker``.
        """

        pool = mp.Pool(processes, **kwargs)
        res = list(tqdm(pool.imap(worker, self), total=len(self)))

        return res

    def append(self, cell_obj):
        """
        Append Cell object `cell_obj` to the list of cells.

        Parameters
        ----------
        cell_obj : :class:`~colicoords.cell.Cell`
            Cell object to append to current cell list.
        """

        assert isinstance(cell_obj, Cell)
        self.cell_list = np.append(self.cell_list, cell_obj)

    def r_dist(self, stop, step, data_name='', norm_x=False, limit_l=None, storm_weight=False, method='gauss',
               sigma=0.3):
        """
        Calculates the radial distribution for all cells of a given data element.

        Parameters
        ----------
        stop : :obj:`float`
            Until how far from the cell spine the radial distribution should be calculated
        step : :obj:`float`
            The binsize of the returned radial distribution
        data_name : :obj:`str`
            The name of the data element on which to calculate the radial distribution
        norm_x : :obj:`bool`
            If `True` the returned distribution will be normalized with the cell's radius set to 1.
        limit_l : :obj:`str`
            If `None`, all datapoints are used. This can be limited by providing the value `full` (omit poles only),
            'poles' (include only poles), or a float value between 0 and 1 which will limit the data points by
            longitudinal coordinate around the midpoint of the cell.
        storm_weight : :obj:`bool`
            Only applicable for analyzing STORM-type data elements. If `True` the returned histogram is weighted with
            the values in the 'Intensity' field.
        method : :obj:`str`, either 'gauss' or 'box'
            Method of averaging datapoints to calculate the final distribution curve.
        sigma : :obj:`float`
            Applies only when `method` is set to 'gauss'. `sigma` gives the width of the gaussian used for convoluting
            datapoints

        Returns
        -------
        xvals : :class:`~numpy.ndarray`
            Array of distances from the cell midline, values are the middle of the bins
        yvals : :class:`~numpy.ndarray`
            2D Array where each row is the bin heights for each cell.
        """

        # todo might be a good idea to warm the user when attempting this on a  list of 3D data
        numpoints = len(np.arange(0, stop + step, step))
        out_arr = np.zeros((len(self), numpoints))
        for i, c in enumerate(self):
            xvals, yvals = c.r_dist(stop, step, data_name=data_name, norm_x=norm_x, storm_weight=storm_weight,
                                    limit_l=limit_l,
                                    method=method, sigma=sigma)
            out_arr[i] = yvals

        return xvals, out_arr

    def l_dist(self, nbins, start=None, stop=None, data_name='', norm_x=True, method='gauss', r_max=None,
               storm_weight=False, sigma=None):
        """
        Calculates the longitudinal distribution of signal for a given data element for all cells.

        Normalization by cell length is enabled by default to remove cell-to-cell variations in length.

        Parameters
        ----------
        nbins : :obj:`int`
            Number of bins between `start` and `stop`.
        start : :obj:`float`
            Distance from `xl` as starting point for the distribution, units are either pixels or normalized units
            if `norm_x=True`.
        stop : :obj:`float`
            Distance from `xr` as end point for the distribution, units are are either pixels or normalized units
            if `norm_x=True`.
        data_name : :obj:`str`
            Name of the data element to use.
        norm_x : :obj:`bool`
            If *True* the output distribution will be normalized.
        r_max : :obj:`float`, optional
            Datapoints within r_max from the cell midline will be included. If `None` the value from the cell's
            coordinate system will be used.
        storm_weight : :obj:`bool`
            If `True` the datapoints of the specified STORM-type data will be weighted by their intensity.
        method : :obj:`str`
            Method of averaging datapoints to calculate the final distribution curve.
        sigma : :obj:`float` or array_like
            Applies only when `method` is set to 'gauss'. `sigma` gives the width of the gaussian used for convoluting
            datapoints. To use a different sigma for each cell `sigma` can be given as a list or array.

        Returns
        -------
        xvals : :class:`~numpy.ndarray`
            Array of distances along the cell midline, values are the middle of the bins/kernel
        yvals : :class:`~numpy.ndarray`
            2D array where every row is the bin heights per cell.
        """

        y_arr = np.zeros((len(self), nbins))
        x_arr = np.zeros((len(self), nbins))
        for i, c in enumerate(self):
            if len(sigma) == len(self):
                _sigma = sigma[i]

            xvals, yvals = c.l_dist(nbins, start=start, stop=stop, data_name=data_name, norm_x=norm_x,
                                    l_mean=self.length.mean(), method=method, r_max=r_max, storm_weight=storm_weight,
                                    sigma=_sigma)
            x_arr[i] = xvals
            y_arr[i] = yvals

        return x_arr, y_arr

    def l_classify(self, data_name=''):
        """
        Classifies foci in STORM-type data by they x-position along the long axis.

        The spots are classified into 3 categories: 'poles', 'between' and 'mid'. The pole category are spots who are to
        the left and right of xl and xr, respectively. The class 'mid' is a section in the middle of the cell with a
        total length of half the cell's length, the class 'between' is the remaining two quarters between 'mid' and
        'poles'

        Parameters
        ----------
        data_name : :obj:`str`
            Name of the STORM-type data element to classify. When its not specified the first STORM data element is used.

        Returns
        -------
        array : :class:`~numpy.ndarray`
            Array of tuples with number of spots in poles, between and mid classes, respectively.
        """

        return np.array([c.l_classify(data_name=data_name) for c in self])

    def a_dist(self):
        raise NotImplementedError()

    def get_intensity(self, mask='binary', data_name='', func=np.mean):
        """
        Returns the fluorescence intensity for each cell.

        Mean fluorescence intensity either in the region masked by the binary image or reconstructed binary image
        derived from the cell's coordinate system. The default return value is the mean fluorescence intensity. Integrated
        intensity can be calculated by using `func=np.sum`.

        Parameters
        ----------
        mask : :obj:`str`
            Either 'binary' or 'coords' to specify the source of the mask used. 'binary' uses the binary image as mask,
            'coords' uses reconstructed binary from coordinate system
        data_name : :obj:`str`
            The name of the image data element to get the intensity values from.
        func : :obj:`callable`
            This function is applied to the data elements pixels selected by the masking operation. The default is
            `np.mean()`.

        Returns
        -------
        value : :obj:`float`
            Mean fluorescence pixel value.
        """

        return np.array([c.get_intensity(mask=mask, data_name=data_name, func=func) for c in self])

    def measure_r(self, data_name='brightfield', mode='max', in_place=True, **kwargs):
        """
        Measure the radius of the cells.

        The radius is found by the intensity-mid/min/max-point of the radial distribution derived from brightfield
        (default) or another data element.

        Parameters
        ----------
        data_name : :obj:`str`
            Name of the data element to use.
        mode : :obj:`str`
            Mode to find the radius. Can be either 'min', 'mid' or 'max' to use the minimum, middle or maximum value
            of the radial distribution, respectively.
        in_place : :obj:`bool`
            If `True` the found value of `r` is directly substituted in the cell's coordinate system, otherwise the
            value is returned.

        Returns
        -------
        radius : :class:`np.ndarray`
            The measured radius `r` values if `in_place` is `False`, otherwise `None`.
        """

        r = [c.measure_r(data_name=data_name, mode=mode, in_place=in_place, **kwargs) for c in self]
        if not in_place:
            return np.array(r)

    def copy(self):
        """
        Make a copy of the `CellList` object and all its associated data elements.

        This is a deep copy meaning that all numpy data arrays are copied in memory and therefore modifying the copied
        cell objects does not modify the original cell objects.

        Returns
        -------
        cell_list : :class:`CellList`:
            Copied `CellList` object
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
        if isinstance(key, numbers.Integral):
            return self.cell_list.__getitem__(key)
        else:
            out = self.__class__.__new__(self.__class__)
            out.cell_list = self.cell_list.__getitem__(key)
            return out

    def __setitem__(self, key, value):
        assert isinstance(value, Cell)
        self.cell_list.__setitem__(key, value)

    def __contains__(self, item):
        return self.cell_list.__contains__(item)


def solve_general(a, b, c, d):
    """
    Solve cubic polynomial in the form a*x^3 + b*x^2 + c*x + d.

    Only works if polynomial discriminant < 0, then there is only one real root which is the one that is returned. [1]_


    Parameters
    ----------
    a : array_like
        Third order polynomial coefficient.
    b : array_like
        Second order polynomial coefficient.
    c : array_like
        First order polynomial coefficient.
    d : array_like
        Zeroth order polynomial coefficient.

    Returns
    -------
    array : array_like
        Real root solution.

    .. [1] https://en.wikipedia.org/wiki/Cubic_function#General_formula

    """

    # todo check type for performance gain?
    # 16 16: 5.03 s
    # 32 32: 3.969 s
    # 64 64: 5.804 s
    # 8 8:
    d0 = b ** 2. - 3. * a * c
    d1 = 2. * b ** 3. - 9. * a * b * c + 27. * a ** 2. * d

    r0 = np.square(d1) - 4. * d0 ** 3.
    r1 = (d1 + np.sqrt(r0)) / 2
    dc = np.cbrt(
        r1)  # power (1/3) gives nan's for coeffs [1.98537881e+01, 1.44894594e-02, 2.38096700e+00]01, 1.44894594e-02, 2.38096700e+00]
    return -(1. / (3. * a)) * (b + dc + (d0 / dc))
    # todo hit a runtimewaring divide by zero on line above once


def solve_trig(a, b, c, d):
    """
    Solve cubic polynomial in the form a*x^3 + b*x^2 + c*x + d
    Only for polynomial discriminant > 0, the polynomial has three real roots [1]_

    Parameters
    ----------
    a : array_like
        Third order polynomial coefficient.
    b : array_like
        Second order polynomial coefficient.
    c : array_like
        First order polynomial coefficient.
    d : array_like
        Zeroth order polynomial coefficient.

    Returns
    -------
    array : array_like
        First real root solution.

    .. [1] https://en.wikipedia.org/wiki/Cubic_function#Trigonometric_solution_for_three_real_roots

    """

    p = (3. * a * c - b ** 2.) / (3. * a ** 2.)
    q = (2. * b ** 3. - 9. * a * b * c + 27. * a ** 2. * d) / (27. * a ** 3.)
    assert (np.all(p < 0))
    k = 0.
    t_k = 2. * np.sqrt(-p / 3.) * np.cos(
        (1 / 3.) * np.arccos(((3. * q) / (2. * p)) * np.sqrt(-3. / p)) - (2 * np.pi * k) / 3.)
    x_r = t_k - (b / (3 * a))
    try:
        assert (np.all(
            x_r > 0))  # dont know if this is guaranteed otherwise boundaries need to be passed and choosing from 3 slns
    except AssertionError:
        pass
        # todo find out if this is bad or not
        # raise ValueError
    return x_r


def calc_lc(xl, xr, coeff):
    """
    Calculate `lc`.

    The returned length is the arc length from `xl` to `xr` integrated along the polynomial p(x) described by `coeff`.

    Parameters
    ----------
    xl : array_like
        Left bound to calculate arc length from. Shape must be compatible with `xl`.
    xr : array_like
        Right bound to calculate arc length to. Shape must be compatible with `xr`.
    coeff : array_like or :obj:`tuple`
        Array or tuple with coordinate polynomial coefficients `a0`, `a1`, `a2`.

    Returns
    -------
    l : array_like
        Calculated length `lc`.
    """

    a0, a1, a2 = coeff
    l = (1 / (4 * a2)) * (
            ((a1 + 2 * a2 * xr) * np.sqrt(1 + (a1 + 2 * a2 * xr) ** 2) + np.arcsinh((a1 + 2 * a2 * xr))) -
            ((a1 + 2 * a2 * xl) * np.sqrt(1 + (a1 + 2 * a2 * xl) ** 2) + np.arcsinh((a1 + 2 * a2 * xl)))
    )

    return l


def solve_length(xr, xl, coeff, length):
    """
    Used to find `xc` in reverse coordinate transformation.

    Function used to find cellular x coordinate `xr` where the arc length from `xl` to `xr` is equal to length given a
    coordinate system with `coeff` as coefficients.

    Parameters
    ----------
    xr : :obj:`float`
        Right boundary x coordinate of calculated arc length.
    xl  : :obj:`float`
        Left boundary x coordinate of calculated arc length.
    coeff : :obj:`list` or :class:`~numpy.ndarray`
        Coefficients a0, a1, a2 describing the coordinate system.
    length : :obj:`float`
        Target length.

    Returns
    -------
    diff : :obj:`float`
        Difference between calculated length and specified length.
    """

    a0, a1, a2 = coeff
    calculated = (1 / (4 * a2)) * (
            ((a1 + 2 * a2 * xr) * np.sqrt(1 + (a1 + 2 * a2 * xr) ** 2) + np.arcsinh((a1 + 2 * a2 * xr))) -
            ((a1 + 2 * a2 * xl) * np.sqrt(1 + (a1 + 2 * a2 * xl) ** 2) + np.arcsinh((a1 + 2 * a2 * xl)))
    )

    return length - calculated


def calc_length(xr, xl, a2, length):
    raise DeprecationWarning()
    a1 = -a2 * (xr + xl)
    l = (1 / (4 * a2)) * (
            ((a1 + 2 * a2 * xr) * np.sqrt(1 + (a1 + 2 * a2 * xr) ** 2) + np.arcsinh((a1 + 2 * a2 * xr))) -
            ((a1 + 2 * a2 * xl) * np.sqrt(1 + (a1 + 2 * a2 * xl) ** 2) + np.arcsinh((a1 + 2 * a2 * xl)))
    )

    return length - l
