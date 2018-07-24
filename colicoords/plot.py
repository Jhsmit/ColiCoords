import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator
import numpy as np
from colicoords.config import cfg
from colicoords import CellList
import seaborn as sns
from scipy import stats
sns.set_style('white')


class CellPlot(object):
    """ Object for plotting single-cell derived data.

    Attributes:
        cell_obj (:class:`Cell`): Single-cell object to plot.
    """
    def __init__(self, cell_obj):
        """

        Args:
            cell_obj (:class:`Cell`): Single-cell object to plot.
        """
        self.cell_obj = cell_obj

    def plot_midline(self, ax=None, **kwargs):
        """Plot the cell's coordinate system midline.

        Args:
            ax (:class:`matplotlib.axes.Axes`:): Optional matplotlib axes to use for plotting.
            **kwargs: Optional kwargs passed to ax.plot().

        Returns:
            :class:`matplotlib.axes.Axes`: The created or specified with `ax` matplotlib axes.

        """
        x = np.linspace(self.cell_obj.coords.xl, self.cell_obj.coords.xr, 100)
        y = self.cell_obj.coords.p(x)
        if 'color' not in kwargs:
            kwargs['color'] = 'r'

        ax = plt.gca() if ax is None else ax
        ax.plot(x, y, **kwargs)
        ymax, xmax = self.cell_obj.data.shape
        ax.set_ylim(ymax, 0)
        ax.set_xlim(0, xmax)
        return ax

    def plot_binary_img(self, ax=None, **kwargs):
        """Plot the cell's binary image. Equivalent to CellPlot.imshow('binary').

        Args:
            ax (:class:`matplotlib.axes.Axes`:): Optional matplotlib axes to use for plotting
            **kwargs: Optional kwargs passed to ax.plot()

        Returns:
            :class:`matplotlib.axes.Axes`: The created or specified with `ax` matplotlib axes

        """

        if 'interpolation' not in kwargs:
            kwargs['interpolation'] = 'nearest'

        ax = plt.gca() if ax is None else ax
        ymax, xmax = self.cell_obj.data.shape
        ax.imshow(self.cell_obj.data.binary_img, extent=[0, xmax, ymax, 0], **kwargs)

        return ax

    def plot_simulated_binary(self, ax=None, **kwargs):
        """Plot the cell's binary image calculated from the coordinate system.

        Args:
            ax (:class:`matplotlib.axes.Axes`:): Optional matplotlib axes to use for plotting.
            **kwargs: Optional kwargs passed to ax.plot().

        Returns:
            :class:`matplotlib.axes.Axes`: The created or specified with `ax` matplotlib axes.

        """

        if 'interpolation' not in kwargs:
            kwargs['interpolation'] = 'nearest'
        img = self.cell_obj.coords.rc < self.cell_obj.coords.r

        ax = plt.gca() if ax is None else ax
        ymax, xmax = self.cell_obj.data.shape
        ax.imshow(img, extent=[0, xmax, ymax, 0], **kwargs)

        return ax

    def plot_bin_fit_comparison(self, ax=None, **kwargs):
        """Plot the cell's binary image together with the calculated binary image from the coordinate system.

        Args:
            ax (:class:`matplotlib.axes.Axes`:): Optional matplotlib axes to use for plotting.
            **kwargs: Optional kwargs passed to ax.plot().

        Returns:
            (:class:`matplotlib.axes.Axes`:): The created or specified with `ax` matplotlib axes.

        """
        if 'interpolation' not in kwargs:
            kwargs['interpolation'] = 'nearest'
        img = self.cell_obj.coords.rc < self.cell_obj.coords.r

        ax = plt.gca() if ax is None else ax
        ymax, xmax = self.cell_obj.data.shape
        ax.imshow(3 - (2 * img + self.cell_obj.data.binary_img), extent=[0, xmax, ymax, 0], **kwargs)

        return ax

    def plot_outline(self, ax=None, **kwargs):
        """Plot the outline of the cell based on the current coordinate system.

        Args:
            ax (:class:`matplotlib.axes.Axes`:): Optional matplotlib axes to use for plotting.
            **kwargs: Optional kwargs passed to ax.plot().

        Returns:
            (:class:`matplotlib.axes.Axes`:): The created or specified with `ax` matplotlib axes.

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
        ax.plot(x_all, y_all, color=color, **kwargs)

        return ax

    def plot_r_dist(self, ax=None, data_name='', norm_x=False, norm_y=False, storm_weight=False, limit_l=None,
                    method='gauss', **kwargs):
        """Plots the radial distribution of a given data element.

        Args:
            ax (:class:`matplotlib.axes.Axes`:): Optional matplotlib axes to use for plotting.
            data_name (:obj:`str`): Name of the data element to use.
            norm_x (:obj:`bool`): If *True* the output distribution will be normalized along the length axis.
            norm_y: (:obj:`bool`): If *True* the output data will be normalized in the y (intensity).
            storm_weight (:obj:`bool`): If *True* the datapoints of the specified STORM-type data will be weighted by their intensity.
            limit_l (:obj:`str`): If `None`, all datapoints are taking into account. This can be limited by providing the
                value `full` (omit poles only), 'poles' (include only poles), or a float value which will limit the data
                points with around the midline where xmid - xlim < x < xmid + xlim.
            **kwargs: Optional kwargs passed to ax.plot().

        Returns:
            (:class:`matplotlib.axes.Axes`:): The created or specified with `ax` matplotlib axes.

        """

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

        if norm_y:
            y /= y.max()

        x = x if norm_x else x * (cfg.IMG_PIXELSIZE / 1000)
        xunits = 'norm' if norm_x else '$\mu m$'
        yunits = 'norm' if norm_y else 'a.u.'

        ax = plt.gca() if ax is None else ax
        ax.plot(x, y, **kwargs)
        ax.set_xlabel('Distance ({})'.format(xunits))
        ax.set_ylabel('Intensity ({})'.format(yunits))
        if norm_y:
            ax.set_ylim(0, 1.1)

        return ax

    def plot_l_dist(self, ax=None, data_name='', r_max=None, norm_x=False, norm_y=False, storm_weight=False,
                    method='gauss', **kwargs):
        #todo refactor to actual l dist! not xc
        """Plots the longitudinal distribution of a given data element.

        Args:
            ax (:class:`matplotlib.axes.Axes`:): Optional matplotlib axes to use for plotting.
            data_name (:obj:`str`): Name of the data element to use.
            r_max: (:obj:`float`): Datapoints within r_max from the cell midline are included. If *None* the value
                from the cell's coordinate system will be used.
            norm_x (:obj:`bool`): If *True* the output distribution will be normalized along the length axis.
            norm_y: (:obj:`bool`): If *True* the output data will be normalized in the y (intensity).
            storm_weight: If *True* the datapoints of the specified STORM-type data will be weighted by their intensity.

        Returns:
            (:class:`matplotlib.axes.Axes`:): The created or specified with `ax` matplotlib axes.

        """
        nbins = kwargs.pop('nbins', cfg.L_DIST_NBINS)
        sigma = kwargs.pop('sigma', cfg.L_DIST_SIGMA)
        x, y = self.cell_obj.l_dist(nbins, data_name=data_name, norm_x=norm_x, r_max=r_max, storm_weight=storm_weight,
                                    method=method, sigma=sigma)
        if norm_y:
            y /= y.max()

        x = x if norm_x else x * (cfg.IMG_PIXELSIZE / 1000)
        xunits = 'norm' if norm_x else '$\mu m$'
        yunits = 'norm' if norm_y else 'a.u.'

        ax = plt.gca() if ax is None else ax
        ax.set_xlabel('Distance ({})'.format(xunits))
        ax.set_ylabel('Intensity ({})'.format(yunits))

        ax = plt.gca() if ax is None else ax
        ax.plot(x, y, **kwargs)
        ax.set_xlabel('Distance ({})'.format(xunits))
        ax.set_ylabel('Intensity ({})'.format(yunits))
        if norm_y:
            ax.set_ylim(0, 1.1)
        else:
            ymin, ymax = ax.get_ylim()
            ax.set_ylim(0, ymax)

        return ax

    def plot_storm(self, data_name='', ax=None, method='plot', bw_method=0.05, upscale=2, alpha_cutoff=None, **kwargs):
        """Graphically represent STORM data

        Args:
            data_name (:obj:`str`): Name of the data element to plot. Must have the data class 'storm'.
            ax (:class:`matplotlib.axes.Axes`:): Optional matplotlib axes to use for plotting.
            method: (:obj:`str`):  Method of visualization. Options are 'plot', 'hist', or 'kde' just plotting points,
                histogram plot or kernel density estimator plot.
            bw_method (:obj:`float`): The method used to calculate the estimator bandwidth. Passed to
                scipy.stats.gaussian_kde.
            upscale: Upscale factor for the output image. Number of pixels is increased wrt data.shape with a factor
                upscale**2
            alpha_cutoff:
            **kwargs: Optional kwargs passed to ax.plot()

        Returns:
            (:class:`matplotlib.axes.Axes`:): The created or specified with `ax` matplotlib axes

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

        x_bins = np.linspace(0, xmax, num=xmax*upscale, endpoint=True)
        y_bins = np.linspace(0, ymax, num=ymax*upscale, endpoint=True)

        h, xedges, yedges = np.histogram2d(x, y, bins=[x_bins, y_bins])

        ax = plt.gca() if ax is None else ax
        if method == 'plot':
            color = kwargs.pop('color', 'r')
            marker = kwargs.pop('marker', '.')
            linestyle = kwargs.pop('linestyle', 'None')
            ax.plot(x, y, color=color, marker=marker, linestyle=linestyle, **kwargs)

        elif method == 'hist':
            cm = plt.cm.get_cmap('Blues')
            cmap = cm if not 'cmap' in kwargs else kwargs.pop('cmap')

            img = h.T
            ax.imshow(img, interpolation='nearest', cmap=cmap, extent=[0, xmax, ymax, 0], **kwargs)
        elif method == 'kde':
            # https://jakevdp.github.io/PythonDataScienceHandbook/05.13-kernel-density-estimation.html
            #todo check the mgrid describes the coords correctly
            X, Y = np.mgrid[0:xmax:xmax*upscale*1j, ymax:0:ymax*upscale*1j]
            positions = np.vstack([X.ravel(), Y.ravel()])
            values = np.vstack([x, y])
            k = stats.gaussian_kde(values, bw_method=bw_method)
            Z = np.reshape(k(positions).T, X.shape)
            img = np.rot90(Z)

            img_norm = img / img.max()
            alphas = np.ones(img.shape)
            if alpha_cutoff:
                alphas[img_norm < 0.3] = img_norm[img_norm < 0.3] / 0.3

            cmap = sns.light_palette("green", as_cmap=True) if not 'cmap' in kwargs else kwargs.pop('cmap')
            normed = Normalize()(img)
            colors = cmap(normed)
            colors[..., -1] = alphas

            ax.imshow(colors, cmap=cmap, extent=[0, xmax, ymax, 0], interpolation='nearest', **kwargs)
        else:
            raise ValueError('Invalid plotting method')

        return ax

    def plot_l_class(self, data_name='', ax=None, **kwargs):
        """Plots a bar chart of how many foci are in a given STORM data set in classes depending on x-position.

        Args:
            data_name (:obj:`str`): Name of the data element to plot. Must have the data class 'storm'.
            ax (:class:`matplotlib.axes.Axes`:): Optional matplotlib axes to use for plotting.
            **kwargs: Optional kwargs passed to ax.bar().

        Returns:
            (:class:`matplotlib.axes.Axes`:): The created or specified with `ax` matplotlib axes

        """
        #todo created in all there return docstrings is not truthful
        cl = self.cell_obj.l_classify(data_name=data_name)

        ax = plt.gca() if ax is None else ax
        ax.bar(np.arange(3), cl, tick_label=['Pole', 'Between', 'Middle'], **kwargs)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_ylabel('Number of spots')

        return ax

    def _plot_storm(self, storm_table, ax=None, kernel=None, bw_method=0.05, upscale=2, alpha_cutoff=None, **kwargs):
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

            cmap = sns.light_palette("green", as_cmap=True) if not 'cmap' in kwargs else kwargs.pop('cmap')
            normed = Normalize()(img)
            colors = cmap(normed)
            colors[..., -1] = alphas

            ax.imshow(colors, cmap=cmap, extent=[0, xmax, ymax, 0], interpolation='nearest', **kwargs)

    def imshow(self, img, ax=None, **kwargs):
        """Equivalent to matplotlib's imshow but with default extent kwargs to assure proper overlay of pixel and
            carthesian coordinates.

        Args:
            img (:obj:`str` or :class:`~numpy.ndarray`) : Image to show. It can be either a data name of the image-type data
                element to plot or a 2D numpy ndarray.
            ax (:class:`matplotlib.axes.Axes`:): Optional matplotlib axes to use for plotting.
            **kwargs: Optional kwargs passed to ax.plot().

        Returns:
            :class:`matplotlib.axes.Axes`: The created or specified with `ax` matplotlib axes

        """
        if type(img) == str:
            img = self.cell_obj.data.data_dict[img]

        xmax = self.cell_obj.data.shape[1]
        ymax = self.cell_obj.data.shape[0]

        extent = kwargs.pop('extent', [0, xmax, ymax, 0])
        interpolation = kwargs.pop('interpolation', 'none')
        cmap = kwargs.pop('cmap', 'viridis')

        ax = plt.gca() if ax is None else ax
        axes_image = ax.imshow(img, extent=extent, interpolation=interpolation, cmap=cmap, **kwargs)
        return axes_image

    @staticmethod
    def figure():
        """Calls matplotlib.pyplot.figure()"""
        return plt.figure()

    @staticmethod
    def show():
        """Calls matplotlib.pyplot.show()"""
        plt.show()

    @staticmethod
    def savefig(*args, **kwargs):
        """Calls matplotlib.pyplot.savefig(*args, **kwargs)"""
        plt.savefig(*args, **kwargs)

    def _plot_intercept_line(self, x_pos, ax=None, **kwargs):
        x = np.linspace(x_pos - 10, x_pos + 10, num=200)
        f_d = self.cell_obj.coords.p_dx(x_pos)
        y = (-x / f_d) + self.cell_obj.coords.p(x_pos) + (x_pos / f_d)

        ax = plt.gca() if ax is None else ax
        ax.plot(x, y, **kwargs)

        return ax


class CellListPlot(object):
    """ Object for plotting single-cell derived data

      Attributes:
          cell_list (:class:`CellList`): List of Cell objects to plot
    """
    def __init__(self, cell_list):
        assert isinstance(cell_list, CellList)
        self.cell_list = cell_list

    def hist_property(self, prop='length', ax=None, **kwargs):
        """Plot a histogram of a given geometrical property.

        Args:
            prop (:obj:`str`): Property to histogram. This can be one of 'length', radius, 'circumference', 'area',
                'surface' or 'volume'.
            ax (:class:`matplotlib.axes.Axes`:): Optional matplotlib axes to use for plotting.
            **kwargs: Optional kwargs passed to ax.hist().

        Returns:
            :class:`matplotlib.axes.Axes`: The created or specified with `ax` matplotlib axes

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
            xlabel = r'Volume ($\mu m^{3}$'
        else:
            raise ValueError('Invalid target')

        ax = plt.gca() if ax is None else ax
        bins = kwargs.pop('bins', 'fd')
        ax.hist(values, bins=bins, **kwargs)

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Cell count')

        return ax

    def hist_intensity(self, mask='binary', data_name='', ax=None, **kwargs):
        """Histogram all cell's mean fluorescence intensity. Intensities values are calculated by calling
        `Cell.get_intensity()`

        Args:
            mask (:obj:`str`): Either 'binary' or 'coords' to specify the source of the mask used
                'binary' uses the binary imagea as mask, 'coords' uses reconstructed binary from coordinate system.
            data_name (:obj:`str`): The name of the image data element to get the intensity values from.
            ax (:class:`matplotlib.axes.Axes`:): Optional matplotlib axes to use for plotting.
            **kwargs: Optional kwargs passed to ax.hist().

        Returns:
            :class:`matplotlib.axes.Axes`: The created or specified with `ax` matplotlib axes.

        """
        values = self.cell_list.get_intensity(mask=mask, data_name=data_name)

        ax = plt.gca() if ax is None else ax
        bins = kwargs.pop('bins', 'fd')
        ax.hist(values, bins=bins, **kwargs)
        ax.set_title('Cell mean fluorescence intensity')
        ax.set_xlabel('Mean fluorescence (a.u.)')
        ax.set_ylabel('Cell count')

        return ax

    #todo put r_dist call kwargs in dedicated dict?
    def plot_r_dist(self, ax=None, data_name='', norm_y=False, norm_x=False, storm_weight=False, limit_l=None,
                    method='gauss', band_func=np.std, **kwargs):
        """Plots the radial distribution of a given data element.

        Args:
            ax (:class:`matplotlib.axes.Axes`:): Optional matplotlib axes to use for plotting.
            data_name (:obj:`str`): Name of the data element to use.
            norm_x (:obj:`bool`): If *True* the output distribution will be normalized along the length axis.
            norm_y: (:obj:`bool`): If *True* the output data will be normalized in the y (intensity).
            storm_weight (:obj:`bool`): If *True* the datapoints of the specified STORM-type data will be weighted by their intensity.
            xlim (:obj:`str`): If `None`, all datapoints are taking into account. This can be limited by providing the
                value `full` (omit poles only), 'poles' (include only poles), or a float value which will limit the data
                points with around the midline where xmid - xlim < x < xmid + xlim.
            band_func (:obj:`callable`): Callable to determine the fill area around the graph. Default is standard deviation.
            **kwargs: Optional kwargs passed to ax.plot().

        Returns:
            (:class:`matplotlib.axes.Axes`:): The created or specified with `ax` matplotlib axes.

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

        if norm_y:
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
        ax.plot(x, mean, **kwargs)

        if band_func:
            width = band_func(out_arr, axis=0)
            ax.fill_between(x, mean + width, mean - width, alpha=0.25)

        ax.set_xlabel('Distance ({})'.format(xunits))
        ax.set_ylabel('Signal intensity ({})'.format(yunits))
        ax.set_title('Radial Distribution')

        if norm_y:
            ax.set_ylim(0, 1.1)

        return ax

    def plot_l_dist(self, ax=None, data_name='', r_max=None, norm_y=False, storm_weight=False, band_func=np.std,
                    method='gauss', **kwargs):
        """Plots the longitudinal distribution of a given data element.

        The data is normalized along the long axis so multiple cells can be combined.

        Args:
            ax (:class:`matplotlib.axes.Axes`:): Optional matplotlib axes to use for plotting.
            data_name (:obj:`str`): Name of the data element to use.
            r_max: (:obj:`float`): Datapoints within r_max from the cell midline are included. If *None* the value
                from the cell's coordinate system will be used.
            norm_y: (:obj:`bool`): If *True* the output data will be normalized in the y (intensity).
            storm_weight (:obj:`bool`): If *True* the datapoints of the specified STORM-type data will be weighted by their intensity.
            band_func (:obj:`callable`): Callable to determine the fill area around the graph. Default is standard deviation.
            **kwargs: Optional kwargs passed to ax.hist().

        Returns:
            (:class:`matplotlib.axes.Axes`:): The created or specified with `ax` matplotlib axes.

        """
        nbins = kwargs.pop('nbins', cfg.L_DIST_NBINS)
        sigma = kwargs.pop('sigma', cfg.L_DIST_SIGMA)
        x_arr, out_arr = self.cell_list.l_dist(nbins, data_name=data_name, norm_x=True, r_max=r_max,
                                               storm_weight=storm_weight, method=method, sigma=sigma)
        x = x_arr[0]

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
        ax.plot(x, mean, **kwargs)

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

        return ax

    def plot_l_class(self, data_name='', ax=None, yerr='std', **kwargs):
        """Plots a bar chart of how many foci are in a given STORM data set in classes depending on x-position.

        Args:
            data_name (:obj:`str`): Name of the data element to plot. Must have the data class 'storm'.
            ax (:class:`matplotlib.axes.Axes`:): Optional matplotlib axes to use for plotting.
            yerr (:obj:`str`): How to calcuated error bars. Can be 'std' or 'sem' for standard deviation or standard
                error of the mean, respectively.
            **kwargs: Optional kwargs passed to ax.bar().

        Returns:
            (:class:`matplotlib.axes.Axes`:): The created or specified with `ax` matplotlib axes

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
            raise ValueError("Invalid valoue for 'yerr', must be either 'std' or 'sem'")

        ax = plt.gca() if ax is None else ax
        ax.bar(np.arange(3), mean, tick_label=['Pole', 'Between', 'Middle'], yerr=yerr, **kwargs)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_ylabel('Mean number of spots')

        return ax

    def show(self):
        """Calls matplotlib.pyplot.show()"""
        plt.show()




