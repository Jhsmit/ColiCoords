import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator
import numpy as np
from colicoords.config import cfg
from colicoords.cell import calc_lc, CellList
import seaborn as sns
from scipy import stats
from tqdm.auto import tqdm

sns.set_style('white')
cmap_default = {'fluorescence': 'viridis', 'binary': 'gray_r', 'brightfield': 'gray'}


class CellPlot(object):
    """
    Object for plotting single-cell derived data.

    Parameters
    ----------
    cell_obj : :class:`~colicoords.cell.Cell`
        Single-cell object to plot.

    Attributes
    ----------
    cell_obj : :class:`~colioords.cell.Cell`
        Single-cell object to plot.
    """
    def __init__(self, cell_obj):
        self.cell_obj = cell_obj

    def plot_midline(self, ax=None, **kwargs):
        """
        Plot the cell's coordinate system midline.

        Parameters
        ----------
        ax : :class:`~matplotlib.axes.Axes`, optional
            Matplotlib axes to use for plotting.
        **kwargs
            Additional kwargs passed to ax.plot().

        Returns
        -------
        line : :class:`~matplotlib.lines.Line2D`
            Matplotlib line artist object

        """
        x = np.linspace(self.cell_obj.coords.xl, self.cell_obj.coords.xr, 100)
        y = self.cell_obj.coords.p(x)
        if 'color' not in kwargs:
            kwargs['color'] = 'r'

        ax = plt.gca() if ax is None else ax
        line, = ax.plot(x, y, **kwargs)
        ymax, xmax = self.cell_obj.data.shape
        ax.set_ylim(ymax, 0)
        ax.set_xlim(0, xmax)
        return line

    def plot_binary_img(self, ax=None, **kwargs):
        """
        Plot the cell's binary image.

        Equivalent to ``CellPlot.imshow('binary')``.

        Parameters
        ----------
        ax : :class:`matplotlib.axes.Axes`, optional
            Optional matplotlib axes to use for plotting
        **kwargs
            Additional kwargs passed to ax.imshow().

        Returns
        -------
        image : :class:`~matplotlib.image.AxesImage`
            Matplotlib image artist object

        """

        if 'interpolation' not in kwargs:
            kwargs['interpolation'] = 'nearest'

        ax = plt.gca() if ax is None else ax
        ymax, xmax = self.cell_obj.data.shape
        cmap = kwargs.pop('cmap', cmap_default['binary'])
        image = ax.imshow(self.cell_obj.data.binary_img, extent=[0, xmax, ymax, 0], cmap=cmap, **kwargs)

        return image

    def plot_simulated_binary(self, ax=None, **kwargs):
        """
        Plot the cell's binary image calculated from the coordinate system.

        Parameters
        ----------
        ax : :class:`matplotlib.axes.Axes`, optional.
            Matplotlib axes to use for plotting.
        **kwargs
            Additional kwargs passed to ax.imshow().

        Returns
        -------
        image : :class:`~matplotlib.image.AxesImage`
            Matplotlib image artist object

        """

        if 'interpolation' not in kwargs:
            kwargs['interpolation'] = 'nearest'
        img = self.cell_obj.coords.rc < self.cell_obj.coords.r

        ax = plt.gca() if ax is None else ax
        ymax, xmax = self.cell_obj.data.shape
        cmap = kwargs.pop('cmap', cmap_default['binary'])
        image = ax.imshow(img, extent=[0, xmax, ymax, 0], cmap=cmap, **kwargs)

        return image

    def plot_bin_fit_comparison(self, ax=None, **kwargs):
        """
        Plot the cell's binary image together with the calculated binary image from the coordinate system.

        Parameters
        ----------
        ax : :class:`matplotlib.axes.Axes`, optional
            Matplotlib axes to use for plotting.
        **kwargs
            Additional kwargs passed to ax.plot().

        Returns
        -------
        image : :class:`~matplotlib.image.AxesImage`
            Matplotlib image artist object.
        """

        if 'interpolation' not in kwargs:
            kwargs['interpolation'] = 'nearest'
        img = self.cell_obj.coords.rc < self.cell_obj.coords.r

        ax = plt.gca() if ax is None else ax
        ymax, xmax = self.cell_obj.data.shape
        image = ax.imshow(3 - (2 * img + self.cell_obj.data.binary_img), extent=[0, xmax, ymax, 0], **kwargs)

        return image

    def plot_outline(self, ax=None, **kwargs):
        """
        Plot the outline of the cell based on the current coordinate system.

        The outline consists of two semicircles and two offset lines to the central parabola.[1]_[2]_

        Parameters
        ----------
        ax : :class:`~matplotlib.axes.Axes`, optional
            Matplotlib axes to use for plotting.
        **kwargs
            Additional kwargs passed to ax.plot().

        Returns
        -------
        line : :class:`~matplotlib.lines.Line2D`
            Matplotlib line artist object.


        .. [1] T. W. Sederberg. "Computer Aided Geometric Design". Computer Aided Geometric Design Course Notes.
           January 10, 2012
        .. [2] Rida T.Faroukia, Thomas W. Sederberg, Analysis of the offset to a parabola, Computer Aided Geometric Design
            vol 12, issue 6, 1995

        """

        # Parametric plotting of offset line
        # http://cagd.cs.byu.edu/~557/text/ch8.pdf
        #
        # Analysis of the offset to a parabola
        # https://doi-org.proxy-ub.rug.nl/10.1016/0167-8396(94)00038-T

        numpoints = 500
        t = np.linspace(self.cell_obj.coords.xl, self.cell_obj.coords.xr, num=numpoints)
        a0, a1, a2 = self.cell_obj.coords.coeff

        x_top = t + self.cell_obj.coords.r * ((a1 + 2 * a2 * t) / np.sqrt(1 + (a1 + 2 * a2 * t) ** 2))
        y_top = a0 + a1*t + a2*(t**2) - self.cell_obj.coords.r * (1 / np.sqrt(1 + (a1 + 2*a2*t)**2))

        x_bot = t + - self.cell_obj.coords.r * ((a1 + 2 * a2 * t) / np.sqrt(1 + (a1 + 2 * a2 * t) ** 2))
        y_bot = a0 + a1*t + a2*(t**2) + self.cell_obj.coords.r * (1 / np.sqrt(1 + (a1 + 2*a2*t)**2))

        #Left semicirlce
        psi = np.arctan(-self.cell_obj.coords.p_dx(self.cell_obj.coords.xl))

        th_l = np.linspace(-0.5*np.pi+psi, 0.5*np.pi + psi, num=200)
        cl_dx = self.cell_obj.coords.r * np.cos(th_l)
        cl_dy = self.cell_obj.coords.r * np.sin(th_l)

        cl_x = self.cell_obj.coords.xl - cl_dx
        cl_y = self.cell_obj.coords.p(self.cell_obj.coords.xl) + cl_dy

        #Right semicircle
        psi = np.arctan(-self.cell_obj.coords.p_dx(self.cell_obj.coords.xr))

        th_r = np.linspace(0.5*np.pi-psi, -0.5*np.pi-psi, num=200)
        cr_dx = self.cell_obj.coords.r * np.cos(th_r)
        cr_dy = self.cell_obj.coords.r * np.sin(th_r)

        cr_x = cr_dx + self.cell_obj.coords.xr
        cr_y = cr_dy + self.cell_obj.coords.p(self.cell_obj.coords.xr)

        x_all = np.concatenate((cl_x[::-1], x_top, cr_x[::-1], x_bot[::-1]))
        y_all = np.concatenate((cl_y[::-1], y_top, cr_y[::-1], y_bot[::-1]))

        ax = plt.gca() if ax is None else ax
        color = 'r' if 'color' not in kwargs else kwargs.pop('color')
        line, = ax.plot(x_all, y_all, color=color, **kwargs)
        #todo check comma

        return line

    def plot_r_dist(self, ax=None, data_name='', norm_x=False, norm_y=False, zero=False, storm_weight=False, limit_l=None,
                    method='gauss', dist_kwargs=None, **kwargs):
        """
        Plots the radial distribution of a given data element.

        Parameters
        ----------
        ax : :class:`~matplotlib.axes.Axes`, optional
            Matplotlib axes to use for plotting.
        data_name : :obj:`str`
            Name of the data element to use.
        norm_x : :obj:`bool`
            If `True` the output distribution will be normalized along the length axis.
        norm_y: : :obj:`bool`
            If `True` the output data will be normalized in the y (intensity).
        zero : :obj:`bool`
            If `True` the output data will be zeroed.
        storm_weight : :obj:`bool`
            If *True* the datapoints of the specified STORM-type data will be weighted by their intensity.
        limit_l : :obj:`str`
            If `None`, all datapoints are used. This can be limited by providing the value `full` (omit poles only),
            'poles' (include only poles), or a float value between 0 and 1 which will limit the data points by
            longitudinal coordinate around the midpoint of the cell.
        method : :obj:`str`
            Method of averaging datapoints to calculate the final distribution curve.
        dist_kwargs : :obj:`dict`
            Additional kwargs to be passed to :meth:`colicoords.cell.Cell.r_dist`
        **kwargs
            Optional kwargs passed to ``ax.plot()``.

        Returns
        -------
        line : :class:`~matplotlib.lines.Line2D`
            Matplotlib line artist object
        """

        # if norm_x:
        #     stop = cfg.R_DIST_NORM_STOP
        #     step = cfg.R_DIST_NORM_STEP
        #     sigma = cfg.R_DIST_NORM_SIGMA
        # else:
        #     stop = cfg.R_DIST_STOP
        #     step = cfg.R_DIST_STEP
        #     sigma = cfg.R_DIST_SIGMA
        #
        #
        #
        # stop = kwargs.pop('stop', stop)
        # step = kwargs.pop('step', step)
        # sigma = kwargs.pop('sigma', sigma)
        # x, y = self.cell_obj.r_dist(stop, step, data_name=data_name, norm_x=norm_x, storm_weight=storm_weight,
        #                             limit_l=limit_l, method=method, sigma=sigma)

        dist_kwargs = dist_kwargs if dist_kwargs is not None else {}
        x, y = self.get_r_dist(norm_x=norm_x, data_name=data_name, limit_l=limit_l,
                               method=method, storm_weight=storm_weight, **dist_kwargs)

        if not data_name:
            try:
                data_elem = list(self.cell_obj.data.flu_dict.values())[0]  # yuck
            except IndexError:
                try:
                    data_elem = list(self.cell_obj.data.storm_dict.values())[0]
                except IndexError:
                    raise IndexError('No valid data element found')
        else:
            data_elem = self.cell_obj.data.data_dict[data_name]

        if zero:
            y -= y.min()

        if norm_y:
            y = y.astype(float) / y.max()

        x = x if norm_x else x * (cfg.IMG_PIXELSIZE / 1000)
        xunits = 'norm' if norm_x else '$\mu m$'
        yunits = 'norm' if norm_y else 'a.u.'

        ax = plt.gca() if ax is None else ax
        line, = ax.plot(x, y, **kwargs)
        ax.set_xlabel('Distance ({})'.format(xunits))
        if data_elem.dclass == 'storm':
            if storm_weight:
                ylabel = 'Total STORM intensity (photons)'
            else:
                ylabel = 'Number of localizations'
        else:
            ylabel = 'Intensity ({})'.format(yunits)

        ax.set_ylabel(ylabel)
        if norm_y:
            ax.set_ylim(0, 1.1)

        return line

    def get_r_dist(self, norm_x=False, data_name='', limit_l=None, method='gauss', storm_weight=False, **kwargs):
        #todo copy of get_r_dist on CellListPlot, make baseclass?
        #used in kymograph plotting
        if norm_x:
            stop = cfg.R_DIST_NORM_STOP
            step = cfg.R_DIST_NORM_STEP
            sigma = cfg.R_DIST_NORM_SIGMA
        else:
            stop = cfg.R_DIST_STOP
            step = cfg.R_DIST_STEP
            sigma = cfg.R_DIST_SIGMA

        stop = kwargs.pop('stop', stop)
        step = kwargs.pop('step', step)
        sigma = kwargs.pop('sigma', sigma)
        x, y = self.cell_obj.r_dist(stop, step, data_name=data_name, norm_x=norm_x, storm_weight=storm_weight,
                                    limit_l=limit_l, method=method, sigma=sigma)

        return x, y

    def plot_l_dist(self, ax=None, data_name='', r_max=None, norm_x=False, norm_y=False, zero=False, storm_weight=False,
                    method='gauss', dist_kwargs=None, **kwargs):
        """
        Plots the longitudinal distribution of a given data element.

        Parameters
        ----------
        ax : :class:`matplotlib.axes.Axes`, optional
            Matplotlib axes to use for plotting.
        data_name : :obj:`str`
            Name of the data element to use.
        r_max: : :obj:`float`
            Datapoints within r_max from the cell midline are included. If *None* the value from the cell's coordinate
            system will be used.
        norm_x : :obj:`bool`
            If `True` the output distribution will be normalized along the length axis.
        norm_y: : :obj:`bool`
            If `True` the output data will be normalized in the y (intensity).
        zero : :obj:`bool`
            If `True` the output data will be zeroed.
        storm_weight : :obj:`bool`
            If `True` the datapoints of the specified STORM-type data will be weighted by their intensity.
        method : :obj:`str`:
            Method of averaging datapoints to calculate the final distribution curve.
        dist_kwargs : :obj:`dict`
            Additional kwargs to be passed to :meth:`~colicoords.cell.Cell.l_dist`

        Returns
        -------
        line : :class:`~matplotlib.lines.Line2D`
            Matplotlib line artist object.
        """

        if not data_name:
            try:
                data_elem = list(self.cell_obj.data.flu_dict.values())[0]  # yuck
            except IndexError:
                try:
                    data_elem = list(self.cell_obj.data.storm_dict.values())[0]
                except IndexError:
                    raise IndexError('No valid data element found')
        else:
            data_elem = self.cell_obj.data.data_dict[data_name]

        nbins = dist_kwargs.pop('nbins', cfg.L_DIST_NBINS)
        scf = self.cell_obj.length if norm_x else 1
        sigma = dist_kwargs.pop('sigma', cfg.L_DIST_SIGMA) / scf

        dist_kwargs = dist_kwargs if dist_kwargs is not None else {}
        x, y = self.cell_obj.l_dist(nbins, data_name=data_name, norm_x=norm_x, r_max=r_max, storm_weight=storm_weight,
                                    method=method, sigma=sigma, **dist_kwargs)

        if zero:
            y -= y.min()

        if norm_y:
            y = y.astype(float) / y.max()

        x = x if norm_x else x * (cfg.IMG_PIXELSIZE / 1000)
        xunits = 'norm' if norm_x else '$\mu m$'
        yunits = 'norm' if norm_y else 'a.u.'

        ax = plt.gca() if ax is None else ax
        ax.set_xlabel('Distance ({})'.format(xunits))
        ax.set_ylabel('Intensity ({})'.format(yunits))

        ax = plt.gca() if ax is None else ax
        line, = ax.plot(x, y, **kwargs)
        ax.set_xlabel('Distance ({})'.format(xunits))

        if data_elem.dclass == 'storm':
            if storm_weight:
                ylabel = 'Total STORM intensity (photons)'
            else:
                ylabel = 'Number of localizations'
        else:
            ylabel = 'Intensity ({})'.format(yunits)
        ax.set_ylabel(ylabel)
        if norm_y:
            ax.set_ylim(0, 1.1)
        else:
            ymin, ymax = ax.get_ylim()
            ax.set_ylim(0, ymax)

        return line

    def plot_storm(self, ax=None, data_name='', method='plot', upscale=5, alpha_cutoff=None, storm_weight=True, sigma=0.25, **kwargs):
        #todo make functions with table and shape and other kwargs?
        """
        Graphically represent STORM data.

        Parameters
        ----------
        ax : :class:`~matplotlib.axes.Axes`
            Optional matplotlib axes to use for plotting.
        data_name : :obj:`str`
            Name of the data element to plot. Must be of data class 'storm'.
        method : :obj:`str`
            Method of visualization. Options are 'plot', 'hist', or 'gauss' just plotting points, histogram plot or
            gaussian kernel plot.
        upscale : :obj:`int`
            Upscale factor for the output image. Number of pixels is increased w.r.t. data.shape with a factor upscale**2
        alpha_cutoff : :obj:`float`
            Values (normalized) below `alpha_cutoff` are transparent, where the alpha is linearly scaled between 0 and
            `alpha_cutoff`
        storm_weight : :obj:`bool`
            If `True` the STORM data points are weighted by their intensity.
        sigma : :obj:`float` or :obj:`string` or :class:`~numpy.ndarray`
            Only applies for method 'gauss'. The value is the sigma which describes the gaussian kernel. If `sigma` is a
            scalar, the same sigma value is used for all data points. If `sigma` is a string it is interpreted as the
            name of the field in the STORM array to use. Otherwise, sigma can be an array with equal length to the
            number of datapoints.
        **kwargs
            Additional kwargs passed to ax.plot() or ax.imshow().

        Returns
        -------
        artist : :class:`~matplotlib.image.AxesImage` or :class:`~matplotlib.lines.Line2D`
            Matplotlib artist object.
        """
        #todo alpha cutoff docstirng and adjustment / testing
        if not data_name:
            storm_table = list(self.cell_obj.data.storm_dict.values())[0]
        else:
            storm_table = self.cell_obj.data.data_dict[data_name]
            assert storm_table.dclass == 'storm'

        x, y = storm_table['x'], storm_table['y']

        if self.cell_obj.data.shape:
            xmax = self.cell_obj.data.shape[1]
            ymax = self.cell_obj.data.shape[0]
        else:
            xmax = int(storm_table['x'].max())
            ymax = int(storm_table['y'].max())

        extent = kwargs.pop('extent', [0, xmax, ymax, 0])
        interpolation = kwargs.pop('interpolation', 'nearest')
        try:
            intensities = storm_table['intensity'] if storm_weight else np.ones_like(x)
        except ValueError:
            print("Warning: The field 'intensity' was not found, all weights are set to one")
            intensities = np.ones_like(x)

        ax = plt.gca() if ax is None else ax
        if method == 'plot':
            color = kwargs.pop('color', 'r')
            marker = kwargs.pop('marker', '.')
            linestyle = kwargs.pop('linestyle', 'None')
            artist, = ax.plot(x, y, color=color, marker=marker, linestyle=linestyle, **kwargs)

        elif method == 'hist':
            x_bins = np.linspace(0, xmax, num=xmax * upscale, endpoint=True)
            y_bins = np.linspace(0, ymax, num=ymax * upscale, endpoint=True)

            h, xedges, yedges = np.histogram2d(x, y, bins=[x_bins, y_bins])

            cm = plt.cm.get_cmap('Blues')
            cmap = cm if not 'cmap' in kwargs else kwargs.pop('cmap')

            img = h.T
            artist = ax.imshow(img, interpolation=interpolation, cmap=cmap, extent=extent, **kwargs)

        elif method == 'gauss':
            if type(sigma) == str:
                sigma = storm_table[sigma]
            elif isinstance(sigma, np.ndarray):
                assert sigma.shape == x.shape
            elif np.isscalar(sigma):
                sigma = sigma*np.ones_like(x)
            else:
                raise ValueError('Invalid sigma')

            step = 1 / upscale
            xi = np.arange(step / 2, xmax, step)
            yi = np.arange(step / 2, ymax, step)

            x_coords = np.repeat(xi, len(yi)).reshape(len(xi), len(yi)).T
            y_coords = np.repeat(yi, len(xi)).reshape(len(yi), len(xi))
            img = np.zeros_like(x_coords)

            pbar = tqdm if len(sigma) > 1500 else lambda i, total=None: i
            for _sigma, _int, _x, _y in pbar(zip(sigma, intensities, x, y), total=len(sigma)):
                img += _int * np.exp(-(((_x - x_coords) / _sigma) ** 2 + ((_y - y_coords) / _sigma) ** 2) / 2)

            img_norm = img / img.max()

            #            np.ma.masked_where(img_norm < alpha_cutoff, img)

            alphas = np.ones(img.shape)
            if alpha_cutoff:
                alphas[img_norm < alpha_cutoff] = img_norm[img_norm < alpha_cutoff] / alpha_cutoff

            cmap = kwargs.pop('cmap', 'viridis')
            cmap = plt.cm.get_cmap(cmap) if type(cmap) == str else cmap

            normed = Normalize()(img)
            colors = cmap(normed)
            colors[..., -1] = alphas
            artist = ax.imshow(colors, cmap=cmap, extent=extent, interpolation=interpolation, **kwargs)

        elif method == 'gauss_old':
            xmax = self.cell_obj.data.shape[1]
            ymax = self.cell_obj.data.shape[0]

            step = 1 / upscale
            xi = np.arange(step / 2, xmax, step)
            yi = np.arange(step / 2, ymax, step)

            xcoords = np.repeat(xi, len(yi)).reshape(len(xi), len(yi)).T
            ycoords = np.repeat(yi, len(xi)).reshape(len(yi), len(xi))

            mx_i, mx_o = np.meshgrid(x, xcoords.flatten())
            my_i, my_o = np.meshgrid(y, ycoords.flatten())

            if type(sigma) == str:
                sigma_arr = storm_table(sigma)
                sigma = sigma_arr[np.newaxis, :]
            elif isinstance(sigma, np.ndarray):
                assert sigma.shape == x.shape
                sigma = sigma[np.newaxis, :]
            elif np.isscalar(sigma):
                pass
            else:
                raise ValueError('Invalid sigma')

            #todo normalization like this or not? (doesnt really matter in the end)
            # res = 1 / (np.sqrt((2 * np.pi)) * sigma ** 2) * np.exp(
            #     - (((mx_i - mx_o) ** 2 / (2 * sigma ** 2)) + ((my_i - my_o) ** 2 / (2 * sigma ** 2)))
            res = np.exp(-(((mx_i - mx_o) ** 2 / (2 * sigma ** 2)) + ((my_i - my_o) ** 2 / (2 * sigma ** 2))))

            if storm_weight:
                res = res*storm_table['intensity'][np.newaxis, :]

            s = np.sum(res, axis=1)
            img = s.reshape(xcoords.shape)
            img_norm = img / img.max()

#            np.ma.masked_where(img_norm < alpha_cutoff, img)

            alphas = np.ones(img.shape)
            if alpha_cutoff:
                alphas[img_norm < alpha_cutoff] = img_norm[img_norm < alpha_cutoff] / alpha_cutoff

            cmap = kwargs.pop('cmap', 'viridis')
            cmap = plt.cm.get_cmap(cmap) if type(cmap) == str else cmap

            normed = Normalize()(img)
            colors = cmap(normed)
            colors[..., -1] = alphas

            artist = ax.imshow(colors, cmap=cmap, extent=extent, interpolation=interpolation, **kwargs)

        else:
            raise ValueError('Invalid plotting method')

        return artist

    def plot_l_class(self, ax=None, data_name='', **kwargs):
        """
        Plots a bar chart of how many foci are in a given STORM data set in classes depending on x-position.

        Parameters
        ----------
        ax : :class:`matplotlib.axes.Axes`, optional
            Matplotlib axes to use for plotting.
        data_name : :obj:`str`
            Name of the data element to plot. Must have the data class 'storm'.
        **kwargs
            Optional kwargs passed to ax.bar().

        Returns
        -------
        container : :class:`~matplotlib.container.BarContainer`
            Container with all the bars.
        """

        cl = self.cell_obj.l_classify(data_name=data_name)

        ax = plt.gca() if ax is None else ax
        container = ax.bar(np.arange(3), cl, tick_label=['Pole', 'Between', 'Middle'], **kwargs)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_ylabel('Number of spots')

        return container

    def _plot_storm(self, storm_table, ax=None, kernel=None, bw_method=0.05, upscale=2, alpha_cutoff=None, **kwargs):
        raise DeprecationWarning("")
        x, y = storm_table['x'], storm_table['y']

        if self.cell_obj.data.shape:
            xmax = self.cell_obj.data.shape[1]
            ymax = self.cell_obj.data.shape[0]
        else:
            xmax = int(storm_table['x'].max())
            ymax = int(storm_table['y'].max())

        x_bins = np.linspace(0, xmax, num=xmax * upscale, endpoint=True)
        y_bins = np.linspace(0, ymax, num=ymax * upscale, endpoint=True)

        h, xedges, yedges = np.histogram2d(x, y, bins=[x_bins, y_bins])

        ax = plt.gca() if ax is None else ax
        if not kernel:
            cm = plt.cm.get_cmap('Blues')
            cmap = cm if not 'cmap' in kwargs else kwargs.pop('cmap')

            img = h.T
            ax.imshow(img, interpolation='nearest', cmap=cmap, extent=[0, xmax, ymax, 0], **kwargs)
        else:
            # https://jakevdp.github.io/PythonDataScienceHandbook/05.13-kernel-density-estimation.html
            # todo check the mgrid describes the coords correctly
            X, Y = np.mgrid[0:xmax:xmax * upscale * 1j, ymax:0:ymax * upscale * 1j]
            positions = np.vstack([X.ravel(), Y.ravel()])
            values = np.vstack([x, y])
            k = stats.gaussian_kde(values, bw_method=bw_method)
            Z = np.reshape(k(positions).T, X.shape)
            img = np.rot90(Z)

            img_norm = img / img.max()
            alphas = np.ones(img.shape)
            if alpha_cutoff:
                alphas[img_norm < 0.3] = img_norm[img_norm < 0.3] / 0.3

            cmap = sns.light_palette("green", as_cmap=True) if not 'cmap' in kwargs else plt.cm.get_cmap(kwargs.pop('cmap'))
            normed = Normalize()(img)
            colors = cmap(normed)
            colors[..., -1] = alphas

            ax.imshow(colors, cmap=cmap, extent=[0, xmax, ymax, 0], interpolation='nearest', **kwargs)

    def plot_kymograph(self, ax=None, mode='r', data_name='', time_factor=1, time_unit='frames', dist_kwargs=None,
                       norm_y=True, aspect=1, **kwargs):
        """
        Plot a kymograph of a chosen axis distribution for a given data element.

        The data element must be a 3D array (t, y, x) where the first axis is the time dimension.

        Parameters
        ----------
        ax : :class:`matplotlib.axes.Axes`, optional
            Matplotlib axes to use for plotting.
        mode : :obj:`str`
            Axis of distribution to plot. Options are 'r', 'l' or 'a'. Currently only 'r' is implemented.
        data_name : :obj:`str`
            Name of the data element to plot. Must be a 3D array
        time_factor : :obj:`float`
            Time factor per frame.
        time_unit : :obj:`str`
            Time unit.
        dist_kwargs : :obj:`dict`
            Additional kwargs passed to the function getting the distribution.
        norm_y : :obj:`bool`
            If `True` the output kymograph is normalized frame-wise.
        aspect : :obj:`float`
            Aspect ratio of output kymograph image.
        **kwargs
            Additional keyword arguments passed to ax.imshow()

        Returns
        -------
        image : :class:`matplotlib.image.AxesImage`
            Matplotlib image artist object
        """

        # if not data_name:
        #     try:
        #         data_elem = list(self.cell_obj.data.flu_dict.values())[0]  # yuck
        #     except IndexError:
        #         try:
        #             data_elem = list(self.cell_obj.data.storm_dict.values())[0]
        #         except IndexError:
        #             raise IndexError('No valid data element found')
        # else:
        #     data_elem = self.cell_obj.data.data_dict[data_name]
        # assert data_elem.ndim == 3

        dist_kwargs = dist_kwargs if dist_kwargs is not None else {}

        if mode == 'r':
            x, arr = self.get_r_dist(data_name=data_name, **dist_kwargs)
            assert arr.ndim == 2
        elif mode == 'l':
            raise NotImplementedError()
            x, arr = self.get_l_dist()
        elif mode == 'a':
            raise NotImplementedError()
        else:
            raise ValueError('Invalid mode')

        return kymograph(x, arr, ax=ax, time_factor=time_factor, time_unit=time_unit,
                         norm_y=norm_y, aspect=aspect, **kwargs)

    def hist_l_storm(self, data_name='', ax=None, norm_x=True, **kwargs):
        """
        Makes a histogram of the longitudinal distribution of localizations.

        Parameters
        ----------
        data_name : :obj:`str`, optional
            Name of the STORM data element to histogram. If omitted, the first STORM element is used.
        ax : :class:`matplotlib.axes.Axes`
            Matplotlib axes to use for plotting.
        norm_x : :obj:`bool`
            Normalizes the longitudinal distribution by dividing by the length of the cell.
        **kwargs
            Additional kwargs passed to `ax.hist()`

        Returns
        -------
        n : :class:`~numpy.ndarray`
            The values of the histogram bins as produced by :func:`~matplotlib.pyplot.hist`
        bins : :class:`~numpy.ndarray`
            The edges of the bins.
        patches : :obj:`list`
            Silent list of individual patches used to create the histogram.

        """
        if not data_name:
            data_name = list(self.cell_obj.data.storm_dict.keys())[0]

        assert self.cell_obj.data.data_dict[data_name].dclass == 'storm'

        storm_table = self.cell_obj.data.data_dict[data_name]
        xp, yp = storm_table['x'], storm_table['y']

        idx_left, idx_right, xc = self.cell_obj.coords.get_idx_xc(xp, yp)
        x_len = calc_lc(self.cell_obj.coords.xl, xc.flatten(), self.cell_obj.coords.coeff)

        if norm_x:
            x_len /= self.cell_obj.length

        ax = plt.gca() if ax is None else ax
        ax.set_xlabel('Distance (norm)')
        ax.set_ylabel('Number of localizations')
        ax.set_title('Longitudinal Distribution')

        bins = kwargs.pop('bins', 'fd')
        return ax.hist(x_len, bins=bins, **kwargs)

    def hist_r_storm(self, data_name='', ax=None, norm_x=True, **kwargs):
        """
        Makes a histogram of the radial distribution of localizations.

        Parameters
        ----------
        data_name : :obj:`str`, optional
            Name of the STORM data element to histogram. If omitted, the first STORM element is used.
        ax : :class:`matplotlib.axes.Axes`
            Matplotlib axes to use for plotting.
        norm_x : :obj:`bool`
            If `True` all radial distances are normalized by dividing by the radius of the individual cells.
        **kwargs
            Additional kwargs passed to `ax.hist()`

        Returns
        -------
        n : :class:`~numpy.ndarray`
            The values of the histogram bins as produced by :func:`~matplotlib.pyplot.hist`
        bins : :class:`~numpy.ndarray`
            The edges of the bins.
        patches : :obj:`list`
            Silent list of individual patches used to create the histogram.

        """
        if not data_name:
            data_name = list(self.cell_obj.data.storm_dict.keys())[0]

        assert self.cell_obj.data.data_dict[data_name].dclass == 'storm'

        r_coords = []
        storm_table = self.cell_obj.data.data_dict[data_name]

        xp, yp = storm_table['x'], storm_table['y']

        r = self.cell_obj.coords.calc_rc(xp, yp)
        if norm_x:
            r /= self.cell_obj.coords.r

        r_coords.append(r)

        ax = plt.gca() if ax is None else ax
        ax.set_xlabel('Distance (norm)')
        ax.set_ylabel('Number of localizations')
        ax.set_title('Radial Distribution')
        bins = kwargs.pop('bins', 'fd')
        h = ax.hist(r, bins=bins, **kwargs)
        ax.set_xlim(0, None)

        return h

    def hist_phi_storm(self, ax=None, data_name='', **kwargs):
        """
        Makes a histogram of the angular distribution of localizations at the poles.

        Parameters
        ----------
        data_name : :obj:`str`, optional
            Name of the STORM data element to histogram. If omitted, the first STORM element is used.
        ax : :class:`matplotlib.axes.Axes`
            Matplotlib axes to use for plotting.
        **kwargs
            Additional kwargs passed to `ax.hist()`

        Returns
        -------
        n : :class:`~numpy.ndarray`
            The values of the histogram bins as produced by :func:`~matplotlib.pyplot.hist`
        bins : :class:`~numpy.ndarray`
            The edges of the bins.
        patches : :obj:`list`
            Silent list of individual patches used to create the histogram.

        """
        if not data_name:
            data_name = list(self.cell_obj.data.storm_dict.keys())[0]

        assert self.cell_obj.data.data_dict[data_name].dclass == 'storm'

        storm_table = self.cell_obj.data.data_dict[data_name]
        xp, yp = storm_table['x'], storm_table['y']
        phi = self.cell_obj.coords.calc_phi(xp, yp)
        bools = (phi == 0.) + (phi == 180.)

        ax = plt.gca() if ax is None else ax

        ax.set_xlabel('Angle (degrees)')
        ax.set_ylabel('Number of localizations')
        ax.set_title('Angular Distribution')
        bins = kwargs.pop('bins', 'fd')
        h = ax.hist(phi[~bools], bins=bins, **kwargs)

        return h

    def imshow(self, img, ax=None, **kwargs):
        """
        Call to matplotlib's imshow.

        Default `extent` keyword arguments is provided to assure proper overlay of pixel and carthesian coordinates.

        Parameters
        ----------
        img : :obj:`str` or :class:`~numpy.ndarray`
            Image to show. It can be either a data name of the image-type data element to plot or a 2D numpy ndarray.
        ax : :class:`matplotlib.axes.Axes`
            Optional matplotlib axes to use for plotting.
        **kwargs:
            Additional kwargs passed to ax.imshow().

        Returns
        -------
        image : :class:`matplotlib.image.AxesImage`
            Matplotlib image artist object.
        """

        if type(img) == str:
            img = self.cell_obj.data.data_dict[img]

        xmax = self.cell_obj.data.shape[1]
        ymax = self.cell_obj.data.shape[0]

        extent = kwargs.pop('extent', [0, xmax, ymax, 0])
        interpolation = kwargs.pop('interpolation', 'none')
        try:
            cmap = kwargs.pop('cmap', cmap_default[img.dclass] if img.dclass else 'viridis')
        except AttributeError:
            cmap = kwargs.pop('cmap', 'viridis')

        ax = plt.gca() if ax is None else ax
        image = ax.imshow(img, extent=extent, interpolation=interpolation, cmap=cmap, **kwargs)
        return image

    @staticmethod
    def figure(*args, **kwargs):
        """Calls :meth:`matplotlib.pyplot.figure`"""
        return plt.figure(*args, **kwargs)

    @staticmethod
    def show(*args, **kwargs):
        """Calls :meth:`matplotlib.pyplot.show`"""
        plt.show(*args, **kwargs)

    @staticmethod
    def savefig(*args, **kwargs):
        """Calls :meth:`matplotlib.pyplot.savefig`"""
        plt.savefig(*args, **kwargs)

    def _plot_intercept_line(self, x_pos, ax=None, **kwargs):
        x = np.linspace(x_pos - 10, x_pos + 10, num=200)
        f_d = self.cell_obj.coords.p_dx(x_pos)
        y = (-x / f_d) + self.cell_obj.coords.p(x_pos) + (x_pos / f_d)

        ax = plt.gca() if ax is None else ax
        ax.plot(x, y, **kwargs)

        return ax


class CellListPlot(object):
    """
    Object for plotting single-cell derived data

    Parameters
    ----------
    cell_list : :class:`~colicoords.cell.CellList`
        ``CellList`` object with ``Cell`` objects to plot.
    """
    def __init__(self, cell_list):
        assert isinstance(cell_list, CellList)
        self.cell_list = cell_list

    def hist_property(self, prop='length', ax=None, **kwargs):
        """
        Plot a histogram of a given geometrical property.

        Parameters
        ----------
        prop : :obj:`str`
            Property to histogram. This can be one of 'length', radius, 'circumference', 'area', 'surface' or 'volume'.
        ax : :class:`~matplotlib.axes.Axes`, optional
            Matplotlib axes to use for plotting.
        **kwargs
            Additional kwargs passed to ax.hist().

        Returns
        -------
        tuple : :obj:`tuple`
            Return value is a tuple with `n`, `bins`, `patches` as returned by :meth:`~matplotlib.pyplot.hist`.
        """

        if prop == 'length':
            values = self.cell_list.length * (cfg.IMG_PIXELSIZE / 1000)
            title = 'Cell length'
            xlabel = r'Length ($\mu m$)'
        elif prop == 'radius':
            values = self.cell_list.radius * (cfg.IMG_PIXELSIZE / 1000)
            title = 'Cell radius'
            xlabel = r'Radius ($\mu m$)'
        elif prop == 'circumference':
            values = self.cell_list.circumference * (cfg.IMG_PIXELSIZE / 1000)
        elif prop == 'area':
            values = self.cell_list.area * (cfg.IMG_PIXELSIZE / 1000)**2
            title = 'Cell area'
            xlabel = r'Area ($\mu m^{2}$)'
        elif prop == 'surface':
            values = self.cell_list.surface * (cfg.IMG_PIXELSIZE / 1000)**2
            title = 'Cell surface'
            xlabel = r'Area ($\mu m^{2}$)'
        elif prop == 'volume':
            values = self.cell_list.volume * (cfg.IMG_PIXELSIZE / 1000) ** 3
            title = 'Cell volume'
            xlabel = r'Volume ($\mu m^{3}$)'
        else:
            raise ValueError('Invalid target')

        ax = plt.gca() if ax is None else ax
        bins = kwargs.pop('bins', 'fd')
        h = ax.hist(values, bins=bins, **kwargs)

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Cell count')

        return h

    def hist_intensity(self, mask='binary', data_name='', ax=None, **kwargs):
        """
        Histogram all cell's mean fluorescence intensity. Intensities values are calculated by calling
        ``Cell.get_intensity()``

        Parameters
        ----------
        mask : :obj:`str`
            Either 'binary' or 'coords' to specify the source of the mask used 'binary' uses the binary images as mask,
            'coords' uses reconstructed binary from coordinate system.
        data_name : :obj:`str`
            The name of the image data element to get the intensity values from.
        ax : :class:`matplotlib.axes.Axes`, optinal
            Matplotlib axes to use for plotting.
        **kwargs
            Additional kwargs passed to ``ax.hist()``.

        Returns
        -------
        tuple : :obj:`tuple`
            Return value is a tuple with `n`, `bins`, `patches` as returned by :meth:`~matplotlib.pyplot.hist`.
        """

        values = self.cell_list.get_intensity(mask=mask, data_name=data_name)

        ax = plt.gca() if ax is None else ax
        bins = kwargs.pop('bins', 'fd')
        n, bins, patches = ax.hist(values, bins=bins, **kwargs)
        ax.set_title('Cell mean fluorescence intensity')
        ax.set_xlabel('Mean fluorescence (a.u.)')
        ax.set_ylabel('Cell count')

        return n, bins, patches

    #todo put r_dist call kwargs in dedicated dict?
    def plot_r_dist(self, ax=None, data_name='', norm_x=False,  norm_y=False, zero=False, storm_weight=False, limit_l=None,
                    method='gauss', band_func=np.std, **kwargs):
        #todo dist_kwargs, -> adjust docs
        """
        Plots the radial distribution of a given data element.

        Parameters
        ----------
        ax : :class:`matplotlib.axes.Axes`
            Optional matplotlib axes to use for plotting.
        data_name : :obj:`str`
            Name of the data element to use.
        norm_x: : :obj:`bool`
            If `True` the output distribution will be normalized along the length axis.
        norm_y: : :obj:`bool`
            If `True` the output data will be normalized in the y (intensity).
        zero : :obj:`bool`
            If `True` the output data will be zeroed.
        storm_weight : :obj:`bool`
            If `True` the datapoints of the specified STORM-type data will be weighted by their intensity.
        limit_l : :obj:`str`
            If `None`, all datapoints are taking into account. This can be limited by providing the value `full`
            (omit poles only), 'poles' (include only poles), or a float value which will limit the data points with
            around the midline where xmid - xlim < x < xmid + xlim.method : :obj:`str`, either 'gauss' or 'box'
        method : :obj:`str`, either 'gauss' or 'box'
            Method of averaging datapoints to calculate the final distribution curve.
        band_func : :obj:`callable`
            Callable to determine the fill area around the graph. Default is standard deviation.
        **kwargs
            Optional kwargs passed to ax.plot().

        Returns
        -------
        :class:`~matplotlib.lines.Line2D`
            Matplotlib line artist object
        """

        if norm_x:
            stop = cfg.R_DIST_NORM_STOP
            step = cfg.R_DIST_NORM_STEP
            sigma = cfg.R_DIST_NORM_STEP
        else:
            stop = cfg.R_DIST_STOP
            step = cfg.R_DIST_STEP
            sigma = cfg.R_DIST_STEP

        stop = kwargs.pop('stop', stop)
        step = kwargs.pop('step', step)
        sigma = kwargs.pop('sigma', sigma)
        x, out_arr = self.cell_list.r_dist(stop, step, data_name=data_name, norm_x=norm_x, storm_weight=storm_weight,
                                           limit_l=limit_l, method=method, sigma=sigma)
        out_arr = np.nan_to_num(out_arr)

        if zero:
            mins = np.min(out_arr, axis=1)
            out_arr -= mins[:, np.newaxis]

        if norm_y:
            #todo test for storm / sparse
            maxes = np.max(out_arr, axis=1)
            bools = maxes != 0
            n = np.sum(~bools)
            if n > 0:
                print("Warning: removed {} curves with maximum zero".format(n))

            out_arr = out_arr[bools]
            a_max = np.max(out_arr, axis=1)
            out_arr = out_arr / a_max[:, np.newaxis]

        x = x if norm_x else x * (cfg.IMG_PIXELSIZE / 1000)

        xunits = 'norm' if norm_x else '$\mu m$'
        yunits = 'norm' if norm_y else 'a.u.'

        ax = plt.gca() if ax is None else ax
        ax.set_xlabel('Distance ({})'.format(xunits))
        ax.set_ylabel('Intensity ({})'.format(yunits))

        if norm_y:
            ax.set_ylim(0, 1.1)

        mean = np.nanmean(out_arr, axis=0)
        line, = ax.plot(x, mean, **kwargs)

        if band_func:
            width = band_func(out_arr, axis=0)
            ax.fill_between(x, mean + width, mean - width, alpha=0.25)

        ax.set_xlabel('Distance ({})'.format(xunits))
        ax.set_ylabel('Signal intensity ({})'.format(yunits))
        ax.set_title('Radial Distribution')

        if norm_y:
            ax.set_ylim(0, 1.1)

        return line

    def get_r_dist(self, norm_x=False, data_name='', limit_l=None, method='gauss', storm_weight=False, **kwargs):
        #todo wrappertje that autofills defaults?
        if norm_x:
            stop = cfg.R_DIST_NORM_STOP
            step = cfg.R_DIST_NORM_STEP
            sigma = cfg.R_DIST_NORM_SIGMA
        else:
            stop = cfg.R_DIST_STOP
            step = cfg.R_DIST_STEP
            sigma = cfg.R_DIST_SIGMA

        stop = kwargs.pop('stop', stop)
        step = kwargs.pop('step', step)
        sigma = kwargs.pop('sigma', sigma)
        x, y = self.cell_list.r_dist(stop, step, data_name=data_name, norm_x=norm_x, storm_weight=storm_weight,
                                     limit_l=limit_l, method=method, sigma=sigma)

        return x, y

    def plot_l_dist(self, ax=None, data_name='', r_max=None, norm_y=False, zero=False, storm_weight=False, band_func=np.std,
                    method='gauss', dist_kwargs=None, **kwargs):
        """
        Plots the longitudinal distribution of a given data element.

        The data is normalized along the long axis to allow the combining of multiple cells with different lenghts.

        Parameters
        ----------
        ax : :class:`matplotlib.axes.Axes`, optional
            Matplotlib axes to use for plotting.
        data_name : :obj:`str`
            Name of the data element to use.
        r_max : :obj:`float`
            Datapoints within `r_max` from the cell midline are included. If `None` the value from the cell's coordinate
            system will be used.
        norm_y : :obj:`bool`
            If `True` the output data will be normalized in the y (intensity).
        zero : :obj:`bool`
            If `True` the output data will be zeroed.
        storm_weight : :obj:`bool`
            If `True` the datapoints of the specified STORM-type data will be weighted by their intensity.
        band_func : :obj:`callable`
            Callable to determine the fill area around the graph. Default is standard deviation.
        method : :obj:`str`, either 'gauss' or 'box'
            Method of averaging datapoints to calculate the final distribution curve.
        dist_kwargs : :obj:`dict`
            Additional kwargs to be passed to :meth:`~colicoords.cell.CellList.l_dist`
        **kwargs
            Optional kwargs passed to ax.plot()

        Returns
        -------
        line :class:`~matplotlib.lines.Line2D`
            Matplotlib line artist object
        """

        dist_kwargs = {} if dist_kwargs is None else dist_kwargs

        nbins = dist_kwargs.pop('nbins', cfg.L_DIST_NBINS)
        sigma = dist_kwargs.pop('sigma', cfg.L_DIST_SIGMA)
        sigma_arr = sigma / self.cell_list.length
        x_arr, out_arr = self.cell_list.l_dist(nbins, data_name=data_name, norm_x=True, r_max=r_max,
                                               storm_weight=storm_weight, method=method, sigma=sigma_arr, **dist_kwargs)
        x = x_arr[0]

        if zero:
            mins = np.min(out_arr, axis=1)
            out_arr -= mins[:, np.newaxis]

        if norm_y:
            maxes = np.max(out_arr, axis=1)
            bools = maxes != 0
            n = np.sum(~bools)
            if n > 0:
                print("Warning: removed {} curves with maximum zero".format(n))

            out_arr = out_arr[bools]
            a_max = np.max(out_arr, axis=1)
            out_arr = out_arr / a_max[:, np.newaxis]

        xunits = 'norm'
        yunits = 'norm' if norm_y else 'a.u.'

        ax = plt.gca() if ax is None else ax
        ax.set_xlabel('Distance ({})'.format(xunits))
        ax.set_ylabel('Intensity ({})'.format(yunits))

        mean = np.nanmean(out_arr, axis=0)
        line, = ax.plot(x, mean, **kwargs)

        if band_func:
            width = band_func(out_arr, axis=0)
            ax.fill_between(x, mean + width, mean - width, alpha=0.25)

        ax.set_xlabel('Distance ({})'.format(xunits))
        ax.set_ylabel('Signal intensity ({})'.format(yunits))
        ax.set_title('Longitudinal Distribution')

        if norm_y:
            ax.set_ylim(0, 1.1)
        else:
            ymin, ymax = ax.get_ylim()
            ax.set_ylim(0, ymax)

        return line

    def plot_l_class(self, data_name='', ax=None, yerr='std', **kwargs):
        """
        Plots a bar chart of how many foci are in a given STORM data set in classes depending on x-position.

        Parameters
        ----------
        data_name : :obj:`str`
            Name of the data element to plot. Must have the data class 'storm'.
        ax : :class:`matplotlib.axes.Axes`
            Matplotlib axes to use for plotting.
        yerr : :obj:`str`
            How to calculated error bars. Can be 'std' or 'sem' for standard deviation or standard error of the mean,
            respectively.
        **kwargs
            Optional kwargs passed to ax.bar().

        Returns
        -------
        axes : :class:`matplotlib.axes.Axes`
            The created or specified with `ax` matplotlib axes

        """
        #todo created in all the return docstrings is not truthful
        cl = self.cell_list.l_classify(data_name=data_name)
        mean = np.mean(cl, axis=0)
        std = np.std(cl, axis=0)
        if yerr == 'std':
            yerr = std
        elif yerr == 'sem':
            yerr = std / np.sqrt(len(cl))
        else:
            raise ValueError("Invalid value for 'yerr', must be either 'std' or 'sem'")

        ax = plt.gca() if ax is None else ax
        ax.bar(np.arange(3), mean, tick_label=['Pole', 'Between', 'Middle'], yerr=yerr, **kwargs)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_ylabel('Mean number of spots')

        return ax

    def plot_kymograph(self, mode='r', data_name='', ax=None, time_factor=1, time_unit='frames', dist_kwargs=None,
                       norm_y=True, aspect=1, **kwargs):
        """
        Plot a kymograph of a chosen axis distribution for a given data element.

        Each cell in the the ``CellList`` represents one point in time, where the first time point is the first cell in
        the list.

        Parameters
        ----------
        ax : :class:`matplotlib.axes.Axes`, optional
            Matplotlib axes to use for plotting.
        mode : :obj:`str`
            Axis of distribution to plot. Options are 'r', 'l' or 'a'. Currently only 'r' is implemented.
        data_name : :obj:`str`
            Name of the data element to plot. Must be a 3D array
        time_factor : :obj:`float`
            Time factor per frame.
        time_unit : :obj:`str`
            Time unit.
        dist_kwargs : :obj:`dict`
            Additional kwargs passed to the function getting the distribution.
        norm_y : :obj:`bool`
            If `True` the output kymograph is normalized frame-wise.
        aspect : :obj:`float`
            Aspect ratio of output kymograph image.
        **kwargs
            Additional keyword arguments passed to ax.imshow()

        Returns
        -------
        image : :class:`matplotlib.image.AxesImage`
            Matplotlib image artist object
        """

        dist_kwargs = dist_kwargs if dist_kwargs is not None else {}

        if mode == 'r':
            x, arr = self.get_r_dist(data_name=data_name, **dist_kwargs)
            assert arr.ndim == 2
        elif mode == 'l':
            raise NotImplementedError()
            x, arr = self.get_l_dist()
        elif mode == 'a':
            raise NotImplementedError()
        else:
            raise ValueError('Invalid mode')

        return kymograph(x, arr, ax=ax, time_factor=time_factor, time_unit=time_unit, norm_y=norm_y, aspect=aspect, **kwargs)

    def hist_l_storm(self, data_name='', ax=None,  **kwargs):
        """
        Makes a histogram of the longitudinal distribution of localizations.

        All cells are normalized by rescaling the longitudinal coordinates by the lenght of the cells. Polar regions
        are normalized by rescaling with the mean of the length of all cells to ensure uniform scaling of polar regions.

        Parameters
        ----------
        data_name : :obj:`str`, optional
            Name of the STORM data element to histogram. If omitted, the first STORM element is used.
        ax : :class:`matplotlib.axes.Axes`
            Matplotlib axes to use for plotting.
        **kwargs
            Additional kwargs passed to ``ax.hist()``

        Returns
        -------
        n : :class:`~numpy.ndarray`
            The values of the histogram bins as produced by :func:`~matplotlib.pyplot.hist`
        bins : :class:`~numpy.ndarray`
            The edges of the bins.
        patches : :obj:`list`
            Silent list of individual patches used to create the histogram.
        """

        if not data_name:
            data_name = list(self.cell_list[0].data.storm_dict.keys())[0]

        assert self.cell_list[0].data.data_dict[data_name].dclass == 'storm'

        l_mean = self.cell_list.length.mean()
        l_coords = []
        for cell_obj in self.cell_list:
            storm_table = cell_obj.data.data_dict[data_name]

            xp, yp = storm_table['x'], storm_table['y']

            idx_left, idx_right, xc = cell_obj.coords.get_idx_xc(xp, yp)
            x_len = calc_lc(cell_obj.coords.xl, xc.flatten(), cell_obj.coords.coeff)

            len_norm = x_len / cell_obj.length
            len_norm[x_len < 0] = x_len[x_len < 0] / l_mean
            len_norm[x_len > cell_obj.length] = ((x_len[x_len > cell_obj.length] - cell_obj.length) / l_mean) + 1

            l_coords.append(len_norm)

        full_l = np.concatenate(l_coords)

        ax = plt.gca() if ax is None else ax

        ax.set_xlabel('Distance (norm)')
        ax.set_ylabel('Number of localizations')
        ax.set_title('Longitudinal Distribution')

        bins = kwargs.pop('bins', 'fd')
        return ax.hist(full_l, bins=bins, **kwargs)

    def hist_r_storm(self, data_name='', ax=None, norm_x=True, **kwargs):
        """
        Makes a histogram of the radial distribution of localizations.

        Parameters
        ----------
        data_name : :obj:`str`, optional
            Name of the STORM data element to histogram. If omitted, the first STORM element is used.
        ax : :class:`matplotlib.axes.Axes`
            Matplotlib axes to use for plotting.
        norm_x : :obj:`bool`
            If `True` all radial distances are normalized by dividing by the radius of the individual cells.
        **kwargs
            Additional kwargs passed to ``ax.hist()``

        Returns
        -------
        n : :class:`~numpy.ndarray`
            The values of the histogram bins as produced by :func:`~matplotlib.pyplot.hist`
        bins : :class:`~numpy.ndarray`
            The edges of the bins.
        patches : :obj:`list`
            Silent list of individual patches used to create the histogram.
        """
        if not data_name:
            data_name = list(self.cell_list[0].data.storm_dict.keys())[0]

        assert self.cell_list[0].data.data_dict[data_name].dclass == 'storm'

        r_coords = []
        for cell_obj in self.cell_list:
            storm_table = cell_obj.data.data_dict[data_name]

            xp, yp = storm_table['x'], storm_table['y']

            r = cell_obj.coords.calc_rc(xp, yp)
            if norm_x:
                r /= cell_obj.coords.r

            r_coords.append(r)

        full_r = np.concatenate(r_coords)

        ax = plt.gca() if ax is None else ax

        ax.set_xlabel('Distance (norm)')
        ax.set_ylabel('Number of localizations')
        ax.set_title('Radial Distribution')
        bins = kwargs.pop('bins', 'fd')
        h = ax.hist(full_r, bins=bins, **kwargs)
        ax.set_xlim(0, None)

        return h

    def hist_phi_storm(self, ax=None, data_name='', **kwargs):
        """
        Makes a histogram of the angular distribution of localizations at the poles.

        Parameters
        ----------
        data_name : :obj:`str`, optional
            Name of the STORM data element to histogram. If omitted, the first STORM element is used.
        ax : :class:`matplotlib.axes.Axes`
            Matplotlib axes to use for plotting.
        **kwargs
            Additional kwargs passed to ``ax.hist()``

        Returns
        -------
        n : :class:`~numpy.ndarray`
            The values of the histogram bins as produced by :func:`~matplotlib.pyplot.hist`.
        bins : :class:`~numpy.ndarray`
            The edges of the bins.
        patches : :obj:`list`
            Silent list of individual patches used to create the histogram.

        """
        if not data_name:
            data_name = list(self.cell_list[0].data.storm_dict.keys())[0]

        assert self.cell_list[0].data.data_dict[data_name].dclass == 'storm'

        phi_coords = []
        for cell_obj in self.cell_list:
            storm_table = cell_obj.data.data_dict[data_name]

            xp, yp = storm_table['x'], storm_table['y']
            phi = cell_obj.coords.calc_phi(xp, yp)
            bools = (phi == 0.) + (phi == 180.)
            phi_coords.append(phi[~bools])

        full_phi = np.concatenate(phi_coords)

        ax = plt.gca() if ax is None else ax

        ax.set_xlabel('Angle (degrees)')
        ax.set_ylabel('Number of localizations')
        ax.set_title('Angular Distribution')
        bins = kwargs.pop('bins', 'fd')
        h = ax.hist(full_phi, bins=bins, **kwargs)

        return h

    @staticmethod
    def show():
        """Calls matplotlib.pyplot.show()"""
        plt.show()

    @staticmethod
    def figure():
        """Calls matplotlib.pyplot.figure()"""
        return plt.figure()

    @staticmethod
    def savefig(*args, **kwargs):
        """Calls matplotlib.pyplot.savefig(*args, **kwargs)"""
        plt.savefig(*args, **kwargs)


def kymograph(x, arr, ax=None, time_factor=1, time_unit='frames', norm_y=True, aspect=1, **kwargs):
    # Mirror array to show symmetrical left and right sides
    # todo when implementing l_list this should be moved to plot_kymograph
    combined = np.concatenate((arr[:, :0:-1], arr), axis=1)

    if norm_y:
        maxes = np.max(combined, axis=1)
        mins = np.min(combined, axis=1)
        norm_y = (combined - mins[:, np.newaxis]) / (maxes - mins)[:, np.newaxis]
    else:
        norm_y = combined

    # mirror x array
    x_full = np.concatenate((-x[:0:-1], x))

    # x array with datapoints equal to y axis
    x_new = np.linspace(np.min(x_full), np.max(x_full), num=norm_y.shape[0], endpoint=True)

    # interpolate values for new x array
    img = np.empty((norm_y.shape[0], norm_y.shape[0]))
    for i, row in enumerate(norm_y):
        img[i] = np.interp(x_new, x_full, row)

    # Change x units
    x_full *= cfg.IMG_PIXELSIZE / 1000

    # Change y units
    y_max = img.shape[0] * time_factor

    x_range = x_full.max() - x_full.min()
    aspect_c = y_max / x_range

    ax = plt.gca() if ax is None else ax
    image = ax.imshow(img, aspect=aspect * (1 / aspect_c), interpolation='spline16', cmap='viridis', origin='lower_left',
              extent=[x_full.min(), x_full.max(), 0, y_max], **kwargs)
    ax.set_xlabel('Distance ($\mu$m)')
    ax.set_ylabel('Time ({})'.format(time_unit))

    return image
