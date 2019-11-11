import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.projections import register_projection
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import Normalize
from ipywidgets import widgets
from IPython.display import display
import numpy as np

from colicoords.cell import CellList, calc_lc
from colicoords.plot import cmap_default, render_storm
from colicoords.support import pad_cell
from colicoords.config import cfg

#todo fix extend!
# Pad all cells so the shape is the same and then use the good old update?


class IterRedrawAxes(Axes):
    """
    Axes object with methods for iterative plotting.

    Upon renewing the graph for all plotted elements is redrawn completely.

    *args
        Arguments passed to :class:`~matplotlib.axes.Axes`
    **kwargs
        Keyword arguments passed to :class:`~matplotlib.axes.Axes`
    """

    name = 'iter_redraw'

    def __init__(self, *args, **kwargs):
        super(IterRedrawAxes, self).__init__(*args, **kwargs)
        self.redraw_register = []

    def iter_plot(self, *args, **kwargs):
        """
        Plot y versus x through :func:`~matplotlib.pyplot.plot` iteratively.

        Parameters
        ----------
        *args
            Arguments in similar format as :func:`~matplotlib.pyplot.plot`, where the all data arguments should be
            interables with equal lengths.
        **kwargs
            Additional keyword arguments passed to :func:`matplotlib.pyplot.plot`
        Returns
        -------
        lines : :obj:`list`
            List with :class:`matplotlib.lines.Line2D` objects from the first element of data iterables.
        """
        # todo allow single x for multiple y's

        self._set_length(len(args[0]))
        args_grps = []
        while args:
            this, args = list(args[:2]), args[2:]
            try:
                if isinstance(args[0], str):
                    this.append(args[0])
                    args = args[1:]
            except IndexError:
                pass
            args_grps.append(this)

        lines = []
        for args in args_grps:
            iter_args = [a for a in args if not isinstance(a, str)]
            other_args = [a for a in args if isinstance(a, str)]
            line, = self.plot(*[a[0] for a in iter_args], *other_args, **kwargs)
            self.redraw_register.append((self.plot, iter_args, other_args, kwargs))
            lines.append(line)
        return lines

    def iter_imshow(self, X, **kwargs):
        """
        Plot an image through :func:`~matplotlib.pyplot.imshow` iteratively.

        Parameters
        ----------
        X : iterable
            Iterable of image data supported by :func:`~matplotlib.pyplot.imshow`.
        **kwargs
            Additional keyword arguments passed to :func:`~matplotlib.pyplot.imshow`.

        Returns
        -------
        img : :class:`matplotlib.image.AxesImage`
            From the first element of iterable X.
        """

        self._set_length(len(X))
        img = self.imshow(X[0], **kwargs)
        self.redraw_register.append((self.imshow, [X], [], kwargs))
        return img

    def iter_hist(self, x, **kwargs):
        """
        Plot a histogram through :func:`~matplotlib.pyplot.hist` iteratively.

        Parameters
        ----------
        x : iterable
            iterable of array like data to histogram.
        **kwargs
            Additional keyword arguments passed to :func:`~matplotlib.pyplot.hist` iteratively.

        Returns
        -------
        n : :class:`~numpy.ndarray` or list of arrays
            The values of the histogram bins for the first iterable
        b : :class:`~numpy.ndarray`
            Bins of the first histogram in the iterable
        p : list or list of lists
            Silent list of individual :class:`~matplotlib.patches.Patch` objects of the first histogram in the iterable
        """

        self._set_length(len(x))
        self.set_prop_cycle(None)
        n, b, p = self.hist(x[0], **kwargs)
        color = p[0].get_facecolor()
        kwargs.update({'color': color})
        self.redraw_register.append((self.hist, [x], [], kwargs))
        return n, b, p

    def update_graph(self, idx):
        """
        Updates the graph to the requested index.

        Parameters
        ----------
        idx : :obj:`int`
            Index of iterable to plot by redrawing each element

        """

        self.cla()
        for f, iter_args, args, kwargs in self.redraw_register:
            f(*[a[idx] for a in iter_args], *args, **kwargs)

    def _set_length(self, length):
        fig = self.get_figure()
        if fig.length is None:
            fig.set_length(length)
        else:
            assert length == fig.length


#TODO thread/process?? per axes for updating the figure?
class IterUpdateAxes(Axes):
    """
    Axes object with methods for iterative plotting.

    Upon renewing the graph for all plotted elements the data is updated, while leaving the original Axes
    unaltered.

    *args
        Arguments passed to :class:`~matplotlib.axes.Axes`
    **kwargs
        Keyword arguments passed to :class:`~matplotlib.axes.Axes`
    """

    name = 'iter_update'

    def __init__(self, *args, **kwargs):
        super(IterUpdateAxes, self).__init__(*args, **kwargs)
        self.update_register = []
        self.new_additions = []

    def iter_plot(self, *args, **kwargs):
        """
        Plot y versus x through :func:`~matplotlib.pyplot.plot` iteratively.

        Parameters
        ----------
        *args
            Arguments in similar format as :func:`~matplotlib.pyplot.plot`, where the all data arguments should be
            interables with equal lengths.
        **kwargs
            Additional keyword arguments passed to :func:`matplotlib.pyplot.plot`

        Returns
        -------
        lines : :obj:`list`
            List with :class:`matplotlib.lines.Line2D` objects from the first element of data iterables.
        """

        # todo allow single x for multiple y's?
        self._set_length(len(args[0]))
        args_grps = []
        while args:
            this, args = list(args[:2]), args[2:]
            try:
                if isinstance(args[0], str):
                    this.append(args[0])
                    args = args[1:]
            except IndexError:
                pass
            args_grps.append(this)

        lines = []
        for args in args_grps:
            line, = self.plot(*[a[0] for a in args[:2]], *args[2:], **kwargs)
            self.update_register.append((self._update_plot, [line, *args[:2]], {}))
            lines.append(line)

        return lines

    def iter_imshow(self, X, **kwargs):
        """
        Plot an image through :func:`~matplotlib.pyplot.imshow` iteratively.

        Parameters
        ----------
        X : iterable
            Iterable of image data supported by :func:`~matplotlib.pyplot.imshow`.
        **kwargs
            Additional keyword arguments passed to :func:`~matplotlib.pyplot.imshow`.

        Returns
        -------
        img : :class:`matplotlib.image.AxesImage`
            From the first element of iterable X.
        """

        self._set_length(len(X))
        img = self.imshow(X[0], **kwargs)
        self.update_register.append((self._update_imshow, [img, X], {}))  # partial?
        return img

    def iter_hist(self, x, **kwargs):
        """
        Plot a histogram through :func:`~matplotlib.pyplot.hist` iteratively.

        Parameters
        ----------
        x : iterable
            iterable of array like data to histogram.
        **kwargs
            Additional keyword arguments passed to :func:`~matplotlib.pyplot.hist`.

        Returns
        -------
        n : :class:`~numpy.ndarray` or list of arrays
            The values of the histogram bins for the first iterable
        b : :class:`~numpy.ndarray`
            Bins of the first histogram in the iterable
        p : list or list of lists
            Silent list of individual :class:`~matplotlib.patches.Patch` objects of the first histogram in the iterable
        """

        self._set_length(len(x))
        self.set_prop_cycle(None)
        n, b, p = self.hist(x[0], **kwargs)
        color = p[0].get_facecolor()
        kwargs.update({'color': color})
        self.update_register.append((self._update_hist, [p, x], kwargs))
        return n, b, p

    def iter_bar(self, x, height, **kwargs):
        """
        Plot a bar chart through :func:`~matplotlib.pyplot.bar` iteratively.

        Parameters
        ----------
        x : iterable
            iterable of bar x positions.
        height : iterable
            iterable of bar heights
        **kwargs
            Additional keyword arguments passed to :func:`~matplotlib.pyplot.bar`.

        Returns
        -------
        bar_container : :class:`~matplotlib.container.BarContainer`
            Matplotlib container for bar plots
        """

        for h in height:
            assert len(x[0] == len(h)), 'All bar heights must have the same length'
        self._set_length(len(x))
        bar_container = self.bar(x[0], height[0], **kwargs)
        self.update_register.append((self._update_bar, [bar_container, height], {}))

        return bar_container

    @staticmethod
    def _update_bar(idx, bar_container, height, **kwargs):
        h = height[idx]
        [rect.set_height(h_elem) for rect, h_elem in zip(bar_container, h)]

    def _update_hist(self, idx, patches, x, **kwargs):
        [p.remove() for p in patches]
        n, b, p = self.hist(x[idx], **kwargs)
        self.new_additions.append((self._update_hist, [p, x], kwargs))

    def _update_plot(self, idx, line, *data):
        if len(data) == 1:
            line.set_ydata(data)
        else:
            x, y = data
            line.set_xdata(x[idx])
            line.set_ydata(y[idx])

    def _update_imshow(self, idx, img, data):
        img.set_data(data[idx])
        img.set_clim(data[idx].min(), data[idx].max())

    def update_graph(self, idx):
        """
        Updates the graph to the requested index.

        Parameters
        ----------
        idx : :obj:`int`
            Index of iterable to plot by redrawing each element

        """
        self.new_additions = []
        remove = []
        for i, (f, args, kwargs) in enumerate(self.update_register):
            f(idx, *args, **kwargs)
            if f == self._update_hist:
                remove.append(i)
        for i in remove:
            del self.update_register[i]
        self.update_register += self.new_additions

    def _set_length(self, length):
        fig = self.get_figure()
        if fig.length is None:
            fig.set_length(length)
        else:
            assert length == fig.length


class IterFigure(Figure):
    """
    Subclass of :class:`matplotlib.figure.Figure` which allows the display of navigation buttons and sliders to
    iterate through data.

    Parameters
    ----------
    *args
        Additional arguments passed to :class:`matplotlib.figure.Figure`
    slider : :obj:`bool`, optional
        Whether or not to show a slider for navigation
    **kwargs:
        Additional keyword arguments passed to :class:`matplotlib.figure.Figure`
    """
    length = None

    def __init__(self, *args, slider=True, **kwargs):
        super(IterFigure, self).__init__(*args, **kwargs)
        self.idx = 0

        self._btn_first = widgets.Button(description='First')
        self._btn_prev = widgets.Button(description='Prev')
        self._int_box = widgets.BoundedIntText(value=0, min=0, max=1)
        self._btn_next = widgets.Button(description='Next')
        self._btn_last = widgets.Button(description='Last')
        self._btn_random = widgets.Button(description='Random')

        self._int_box.observe(self.handle_int_box, names='value')

        self._btn_first.on_click(self.on_first)
        self._btn_prev.on_click(self.on_prev)
        self._btn_next.on_click(self.on_next)
        self._btn_last.on_click(self.on_last)
        self._btn_random.on_click(self.on_random)

        self.btn_hbox = widgets.HBox()
        self.btn_hbox.children = [self._btn_first, self._btn_prev, self._int_box,
                                  self._btn_next, self._btn_last, self._btn_random]

        if slider:
            self._slider = widgets.IntSlider(value=0, min=0, max=1, layout=dict(width='99%'), readout=False)
            widgets.jslink((self._int_box, 'value'), (self._slider, 'value'))

            self.vbox = widgets.VBox()
            self.vbox.children = [self.btn_hbox, self._slider]
            self.box = self.vbox
        else:
            self.box = self.btn_hbox

    def display(self):
        """
        Displays the widgets box with navigation buttons and optionally slider.
        """
        display(self.box)

    def set_length(self, length):
        """
        Sets if the length of the iterable plotted.

        Parameters
        ----------
        length : :obj:`int`
            Length of the iterable
        """

        self.length = length
        self._int_box.max = length - 1
        try:
            self._slider.max = length - 1
        except AttributeError:
            pass

    def handle_int_box(self, change):
        """
        Handler function for changes in the value display in the text box. Updates the `idx` value and triggers
        updating of the graph.
        """
        self.idx = change.new
        self.update_graph()

    def on_first(self, b):
        self._int_box.value = 0

    def on_prev(self, b):
        val = self.idx - 1
        self._int_box.value = val #= 0 if val < 0 else val

    def on_next(self, b):
        val = self.idx + 1
        self._int_box.value = val # self.length - 1 if val >= self.length else val

    def on_last(self, b):
        self._int_box.value = self.length - 1

    def on_random(self, b):
        self._int_box.value = np.random.randint(0, self.length)

    def update_graph(self):
        """
        Updates all axes in the figure to reflect the new state of `idx`.
        """
        for ax in self.axes:
            ax.update_graph(self.idx)
            ax.relim()
            ax.autoscale_view()
        plt.draw()


class IterCellPlot(object):
    """
    Object for plotting single-cell derived data by iterating through a :class:`~colicoords.cell.CellList` object.

    Parameters
    ----------
    cell_list : :class:`~colicoords.cell.CellList`
        CellList object to plot

    pad : :obj:`bool`, optional
        If `True` all cells will be padded to the shape of the largest cell in the `cell_list`. This allows the usage of
        :class:`~colicoords.iplot.IterUpdateAxes` over :class:`~colicoords.iplot.IterRedrawAxes`, which updates faster.

    Attributes
    ----------
    cell_list : :class:`~colicoords.cell.CellList`
        CellList object to plot
    """

    def __init__(self, cell_list, pad=True):
        if pad:
            shape_0, shape_1 = zip(*[cell_obj.data.shape for cell_obj in cell_list])
            self.cell_list = CellList([pad_cell(cell_obj, (np.max(shape_0), np.max(shape_1))) for cell_obj in cell_list])
        else:
            self.cell_list = cell_list

    def plot_midline(self, ax=None, **kwargs):
        """
        Plot the cell's coordinate system midline.

        Parameters
        ---------
        ax : :class:`~matplotlib.axes.Axes`, optional
            Matplotlib axes to use for plotting.
        **kwargs
            Additional kwargs passed to ax.plot().

        Returns
        -------
        line : :class:`~matplotlib.lines.Line2D`
            Matplotlib line artist object

        """
        x = [np.linspace(cell_obj.coords.xl, cell_obj.coords.xr, 100) for cell_obj in self.cell_list]
        y = [cell_obj.coords.p(xi) for xi, cell_obj in zip(x, self.cell_list)]
        if 'color' not in kwargs:
            kwargs['color'] = 'r'

        ax = plt.gca() if ax is None else ax
        line, = ax.iter_plot(x, y, **kwargs)
        ymax, xmax = self.cell_list[0].data.shape
        ax.set_ylim(ymax, 0)
        ax.set_xlim(0, xmax)
        return line

    def plot_binary_img(self, ax=None, **kwargs):
        """
        Plot the cell's binary image.

        Equivalent to CellPlot.imshow('binary').

        Parameters
        ---------
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
        ymax, xmax = self.cell_list[0].data.shape
        cmap = kwargs.pop('cmap', cmap_default['binary'])
        images = [cell_obj.data.binary_img for cell_obj in self.cell_list]
        image = ax.iter_imshow(images, cmap=cmap, extent=[0, xmax, ymax, 0], **kwargs)

        return image

    def plot_simulated_binary(self, ax=None, **kwargs):
        """
        Plot the cell's binary image calculated from the coordinate system.

        Parameters
        ---------
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
        images = [cell_obj.coords.rc < cell_obj.coords.r for cell_obj in self.cell_list]

        ax = plt.gca() if ax is None else ax
        ymax, xmax = self.cell_list[0].data.shape
        cmap = kwargs.pop('cmap', cmap_default['binary'])
        image = ax.iter_imshow(images, extent=[0, xmax, ymax, 0], cmap=cmap, **kwargs)

        return image

    def plot_bin_fit_comparison(self, ax=None, **kwargs):
        """
        Plot the cell's binary image together with the calculated binary image from the coordinate system.

        Parameters
        ---------
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

        images = [3 - (2 * (cell_obj.coords.rc < cell_obj.coords.r) + cell_obj.data.binary_img) for cell_obj in self.cell_list]

        ax = plt.gca() if ax is None else ax
        ymax, xmax = self.cell_list[0].data.shape
        image = ax.iter_imshow(images, extent=[0, xmax, ymax, 0], **kwargs)

        return image

    def plot_outline(self, ax=None, **kwargs):
        """
        Plot the outline of the cell based on the current coordinate system.

        The outline consists of two semicircles and two offset lines to the central parabola.[1]_[2]_

        Parameters
        ---------
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

        x_all, y_all = zip(*[make_outline(cell_obj) for cell_obj in self.cell_list])

        ax = plt.gca() if ax is None else ax
        color = 'r' if 'color' not in kwargs else kwargs.pop('color')
        line = ax.iter_plot(x_all, y_all, color=color, **kwargs)

        return line

    def plot_r_dist(self, ax=None, data_name='', norm_x=False, norm_y=False, zero=False, storm_weight=False, limit_l=None,
                    method='gauss', dist_kwargs=None, **kwargs):
        """
        Plots the radial distribution of a given data element.

        Parameters
        ---------
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
        dist_kwargs : :obj:`dict
            Additional kwargs to be passed to :meth:`~colicoords.cell.Cell.r_dist`
        **kwargs
            Optional kwargs passed to ax.plot().

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
        x, out_arr = self.get_r_dist(norm_x=norm_x, data_name=data_name, limit_l=limit_l,
                                     method=method, storm_weight=storm_weight, **dist_kwargs)

        if not data_name:
            try:
                data_elem = list(self.cell_list[0].data.flu_dict.values())[0]  # yuck
            except IndexError:
                try:
                    data_elem = list(self.cell_list.data.storm_dict.values())[0]
                except IndexError:
                    raise IndexError('No valid data element found')
        else:
            data_elem = self.cell_list[0].data.data_dict[data_name]

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
        line, = ax.iter_plot([x for i in range(len(self.cell_list))], out_arr, **kwargs)
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

        dist_kwargs = {} if dist_kwargs is None else dist_kwargs
        if not data_name:
            try:
                data_elem = list(self.cell_list[0].data.flu_dict.values())[0]  # yuck
            except IndexError:
                try:
                    data_elem = list(self.cell_list[0].data.storm_dict.values())[0]
                except IndexError:
                    raise IndexError('No valid data element found')
        else:
            data_elem = self.cell_list[0].data.data_dict[data_name]

        nbins = dist_kwargs.pop('nbins', cfg.L_DIST_NBINS)
        sigma = dist_kwargs.pop('sigma', cfg.L_DIST_SIGMA)
        scf = self.cell_list.length if norm_x else np.ones(len(self.cell_list))

        sigma_arr = sigma / scf

        x_arr, out_arr = self.cell_list.l_dist(nbins, data_name=data_name, norm_x=True, r_max=r_max,
                                               storm_weight=storm_weight, method=method, sigma=sigma_arr, **dist_kwargs)

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

        x_arr = x_arr if norm_x else x_arr * (cfg.IMG_PIXELSIZE / 1000)
        xunits = 'norm' if norm_x else '$\mu m$'
        yunits = 'norm' if norm_y else 'a.u.'

        ax = plt.gca() if ax is None else ax
        ax.set_xlabel('Distance ({})'.format(xunits))
        ax.set_ylabel('Intensity ({})'.format(yunits))

        ax = plt.gca() if ax is None else ax
        line, = ax.iter_plot(x_arr, out_arr, **kwargs)

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
            pass
            #todo relimiting needs to be fixed
            # ymin, ymax = ax.get_ylim()
            # ax.set_ylim(0, ymax)

        return line

    def plot_phi_dist(self, ax=None, data_name='', r_max=None, r_min=0, storm_weight=False, method='gauss',
                      dist_kwargs=None, **kwargs):
        step = kwargs.pop('step', cfg.PHI_DIST_STEP)
        sigma = kwargs.pop('sigma', cfg.PHI_DIST_SIGMA)
        dist_kwargs = {} if dist_kwargs is None else dist_kwargs

        if not data_name:
            try:
                data_elem = list(self.cell_list[0].data.flu_dict.values())[0]  # yuck
            except IndexError:
                try:
                    data_elem = list(self.cell_list[0].data.storm_dict.values())[0]
                except IndexError:
                    raise IndexError('No valid data element found')
        else:
            data_elem = self.cell_list[0].data.data_dict[data_name]

        x_vals, phi_l, phi_r = self.cell_list.phi_dist(step, data_name=data_name, r_max=r_max, r_min=r_min,
                                                      storm_weight=storm_weight, sigma=sigma, method=method, **dist_kwargs)

        ax = plt.gca() if ax is None else ax
        if data_elem.dclass == 'storm':
            if storm_weight:
                ylabel = 'Total STORM intensity (photons)'
            else:
                ylabel = 'Number of localizations'
        else:
            ylabel = 'Intensity (a.u.)'

        ax.set_xlabel('Distance ({})'.format('degrees'))
        ax.set_ylabel(ylabel)

        l = kwargs.pop('label', None)

        line_l, = ax.iter_plot([x_vals for x in range(len(self.cell_list))], phi_l)
        line_r, = ax.iter_plot([x_vals for x in range(len(self.cell_list))], phi_r)

        return line_l, line_r


    def plot_storm(self, ax=None, data_name='', method='plot', upscale=5, alpha_cutoff=None, storm_weight=False, sigma=0.25, **kwargs):
        #todo make functions with table and shape and other kwargs?
        """
        Graphically represent STORM data.

        Parameters
        ---------
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
            Additional kwargs passed to ax.plot() or ax.imshow()

        Returns
        -------
        artist :class:`~matplotlib.image.AxesImage` or :class:`~matplotlib.lines.Line2D`
            Matplotlib artist object.
        """
        #todo alpha cutoff docstirng and adjustment / testing

        if not data_name:
            #todo update via CellListData
            data_name = list(self.cell_list[0].data.storm_dict.keys())[0]

        storm_table = self.cell_list[0].data.data_dict[data_name]
        assert storm_table.dclass == 'storm'

        x, y = storm_table['x'], storm_table['y']

        if self.cell_list.data.shape is not None:
            xmax = self.cell_list.data.shape[1]
            ymax = self.cell_list.data.shape[0]
        else:
            #todo change to global x, y max and not local
            xmax = int(storm_table['x'].max())
            ymax = int(storm_table['y'].max())

        extent = kwargs.pop('extent', [0, xmax, ymax, 0])
        interpolation = kwargs.pop('interpolation', 'nearest')

        ax = plt.gca() if ax is None else ax
        if method == 'plot':
            color = kwargs.pop('color', 'r')
            marker = kwargs.pop('marker', '.')
            linestyle = kwargs.pop('linestyle', 'None')
            x, y = zip(*[(cell.data.data_dict[data_name]['x'], cell.data.data_dict[data_name]['y']) for cell in self.cell_list])
            artist, = ax.iter_plot(x, y, color=color, marker=marker, linestyle=linestyle, **kwargs)

        elif method == 'hist':
            x_bins = np.linspace(0, xmax, num=xmax * upscale, endpoint=True)
            y_bins = np.linspace(0, ymax, num=ymax * upscale, endpoint=True)

            img = np.empty((len(self.cell_list), ymax * upscale - 1, xmax * upscale - 1))
            for i, cell in enumerate(self.cell_list):
                storm_table = cell.data.data_dict[data_name]
                x, y = storm_table['x'], storm_table['y']
                h, xedges, yedges = np.histogram2d(x, y, bins=[x_bins, y_bins])
                img[i] = h.T

            cm = plt.cm.get_cmap('Blues')
            cmap = cm if not 'cmap' in kwargs else kwargs.pop('cmap')

            artist = ax.iter_imshow(img, interpolation=interpolation, cmap=cmap, extent=extent, **kwargs)

        elif method == 'gauss':
            step = 1 / upscale
            xi = np.arange(step / 2, xmax, step)
            yi = np.arange(step / 2, ymax, step)

            x_coords = np.repeat(xi, len(yi)).reshape(len(xi), len(yi)).T
            y_coords = np.repeat(yi, len(xi)).reshape(len(yi), len(xi))

            cmap = kwargs.pop('cmap', 'viridis')
            cmap = plt.cm.get_cmap(cmap) if type(cmap) == str else cmap

            colors_stack = np.empty((len(self.cell_list), *x_coords.shape, 4))
            for i, cell in enumerate(self.cell_list):
                storm_table = cell.data.data_dict[data_name]
                x, y = storm_table['x'], storm_table['y']

                if type(sigma) == str:
                    sigma_local = storm_table[sigma]
                elif isinstance(sigma, np.ndarray):
                    assert sigma.shape == x.shape
                    sigma_local = sigma
                elif np.isscalar(sigma):
                    sigma_local = sigma * np.ones_like(x)
                else:
                    raise ValueError('Invalid sigma')


                try:
                    intensities = storm_table['intensity'] if storm_weight else np.ones_like(x)
                except ValueError:
                    intensities = np.ones_like(x)

                # Make empty image and iteratively add gaussians for each localization
                #img = np.zeros_like(x_coords)

                img = render_storm(x_coords, y_coords, sigma_local, intensities, x, y)

                # @jit(nopython=True)
                # for _sigma, _int, _x, _y in zip(sigma_local, intensities, x, y):
                #         img += _int * np.exp(-(((_x - x_coords) / _sigma) ** 2 + ((_y - y_coords) / _sigma) ** 2) / 2)

                img_norm = img / img.max()
                alphas = np.ones(img.shape)
                if alpha_cutoff:
                    alphas[img_norm < alpha_cutoff] = img_norm[img_norm < alpha_cutoff] / alpha_cutoff

                normed = Normalize()(img)
                colors = cmap(normed)
                colors[..., -1] = alphas

                colors_stack[i] = colors

            artist = ax.iter_imshow(colors_stack, cmap=cmap, extent=extent, interpolation=interpolation, **kwargs)

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

        #cl = [cell_obj.l_classify(data_name=data_name) for cell_obj in self.cell_list]

        ax = plt.gca() if ax is None else ax
        container = ax.iter_bar([np.arange(3) for _ in range(len(self.cell_list))], self.cell_list.l_classify(data_name=data_name),
                           tick_label=['Pole', 'Between', 'Middle'], **kwargs)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_ylabel('Number of spots')

        return container

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
            data_name = list(self.cell_list[0].data.storm_dict.keys())[0]

        assert self.cell_list[0].data.data_dict[data_name].dclass == 'storm'

        l_coords = []
        for cell_obj in self.cell_list:
            storm_table = cell_obj.data.data_dict[data_name]

            xp, yp = storm_table['x'], storm_table['y']

            idx_left, idx_right, xc = cell_obj.coords.get_idx_xc(xp, yp)
            x_len = calc_lc(cell_obj.coords.xl, xc.flatten(), cell_obj.coords.coeff)

            if norm_x:
                x_len /= cell_obj.length

            l_coords.append(x_len)

        ax = plt.gca() if ax is None else ax
        ax.set_xlabel('Distance (norm)')
        ax.set_ylabel('Number of localizations')
        ax.set_title('Longitudinal Distribution')

        bins = kwargs.pop('bins', 'fd')
        return ax.iter_hist(l_coords, bins=bins, **kwargs)

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

        ax = plt.gca() if ax is None else ax

        ax.set_xlabel('Distance (norm)')
        ax.set_ylabel('Number of localizations')
        ax.set_title('Radial Distribution')
        bins = kwargs.pop('bins', 'fd')
        h = ax.iter_hist(r_coords, bins=bins, **kwargs)
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
            data_name = list(self.cell_list[0].data.storm_dict.keys())[0]

        assert self.cell_list[0].data.data_dict[data_name].dclass == 'storm'

        phi_coords = []
        for cell_obj in self.cell_list:
            storm_table = cell_obj.data.data_dict[data_name]

            xp, yp = storm_table['x'], storm_table['y']
            phi = cell_obj.coords.calc_phi(xp, yp)
            bools = (phi == 0.) + (phi == 180.)
            phi_coords.append(phi[~bools])

        ax = plt.gca() if ax is None else ax

        ax.set_xlabel('Angle (degrees)')
        ax.set_ylabel('Number of localizations')
        ax.set_title('Angular Distribution')
        bins = kwargs.pop('bins', 'fd')
        h = ax.iter_hist(phi_coords, bins=bins, **kwargs)

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
            cmap = kwargs.pop('cmap', cmap_default[self.cell_list[0].data.data_dict[img].dclass])
            img = np.stack([cell_obj.data.data_dict[img] for cell_obj in self.cell_list])
        else:
            cmap = kwargs.pop('cmap', 'viridis')
        assert img.ndim == 3

        xmax = self.cell_list[0].data.shape[1]
        ymax = self.cell_list[0].data.shape[0]

        extent = kwargs.pop('extent', [0, xmax, ymax, 0])
        interpolation = kwargs.pop('interpolation', 'none')
        # print(img[0].dclass)
        # print(cmap_default[img[0].dclass])
        # try:
        #     cmap = kwargs.pop('cmap', cmap_default[img[0].dclass] if img[0].dclass else 'viridis')
        # except AttributeError:
        #     cmap = kwargs.pop('cmap', 'viridis')

        ax = plt.gca() if ax is None else ax
        image = ax.iter_imshow(img, extent=extent, interpolation=interpolation, cmap=cmap, **kwargs)
        return image

    @staticmethod
    def show(*args, **kwargs):
        """Calls :meth:`matplotlib.pyplot.show`"""
        plt.show(*args, **kwargs)

    @staticmethod
    def savefig(*args, **kwargs):
        """Calls :meth:`matplotlib.pyplot.savefig`"""
        plt.savefig(*args, **kwargs)


class AutoIterCellPlot(IterCellPlot):
    """
    Quickly provides insight into the contents of a :class:`~colicoords.cell.CellList` object by automatically plotting
    the contents of the cells' :class:`~colicoords.data_models.Data` objects.

    Parameters
    ----------
    cell_list : :class:`~colicoords.cell.CellList`
        CellList object to plot

    pad : :obj:`bool`, optional
        If `True` all cells will be padded to the shape of the largest cell in the `cell_list`. This allows the usage of
        :class:`~colicoords.iplot.IterUpdateAxes` over :class:`~colicoords.iplot.IterRedrawAxes`, which updates faster.
    """

    default_order = {
        'binary': 0,
        'brightfield': 1,
        'fluorescence': 2,
        'storm': 3
    }

    def __init__(self, cell_list, pad=True):
        super(AutoIterCellPlot, self).__init__(cell_list, pad=pad)

        #check equal data object for all cells? or assume?
        self.dclasses = self.cell_list[0].data.dclasses
        self.names = self.cell_list[0].data.names
        for c in self.cell_list:
            assert set(self.dclasses) == set(c.data.dclasses), 'All cell must have equal data elements'

        self.num_img = len([d for d in self.dclasses if d != 'storm'])

        self.fig = None
        self.axes = None

    def plot(self, cols=3, **kwargs):
        """
        Creates a :class:`~matplotlib.figure.Figure` by calling :func:`~colicoords.iplot.iter_subplots` and plots all
        data elements associated with the cells.

        Parameters
        ----------
        cols: :obj:`int`
            Number of columns to use in the subplot
        **kwargs
            Additional keyword arguments to pass to :func:`~colicoords.iplot.iter_subplots`.

        """
        rows = int(np.ceil(self.num_img / cols))
        cols = min(cols, self.num_img)

        figsize = kwargs.pop('figsize', (9.5, 3))
        self.fig, self.axes = iter_subplots(rows, cols, figsize=figsize, **kwargs)
        names, dclasses = zip(*[pair for pair in sorted(zip(self.names, self.dclasses), key=lambda pair: self.default_order[pair[1]])])

        for ax, name, dclass in zip(self.axes.flatten(), names, dclasses):
            if dclass == 'storm':
                self.plot_storm(ax=ax, method='gauss')
                self.plot_storm(ax=ax, alpha=0.5)
            else:
                self.imshow(name, ax=ax)
                self.plot_outline(ax=ax, alpha=0.5)
            ax.set_title(name)

        plt.tight_layout()
        self.fig.display()


def iter_subplots(*args, **kwargs):
    """
    Equivalent to :func:`~matplotlib.pyplot.subplots` but by default returns :class:`~colicoords.iplot.IterUpdateAxes`

    Parameters
    ----------
    args
    kwargs

    Returns
    -------
    tuple : :obj:`tuple`
        Tuple containing a :class:`~matplotlib.figure.Figure` and axes as either a single Axes instance or
        :obj:`tuple` with multiple Axes instances
    """
    subplot_kw = kwargs.pop('subplot_kw', {'projection': 'iter_update'})

    fig, axes = plt.subplots(*args,  subplot_kw=subplot_kw, FigureClass=IterFigure, **kwargs)
    return fig, axes


def make_outline(cell_obj):
    """
    Calculates the iso-distance line from the cell's midline, as defined by the cells coordinate system parameters.

    Parameters
    ----------
    cell_obj : :class:`~colicoords.cell.Cell`
        Cell object to base the outline on

    Returns
    -------
    x_all : :class:`~numpy.ndarray`
        x points describing the outline
    y_all : :class:`~numpy.ndarray`
        y points describing the outline
    """

    t = np.linspace(cell_obj.coords.xl, cell_obj.coords.xr, num=500)
    a0, a1, a2 = cell_obj.coords.coeff

    x_top = t + cell_obj.coords.r * ((a1 + 2 * a2 * t) / np.sqrt(1 + (a1 + 2 * a2 * t) ** 2))
    y_top = a0 + a1*t + a2*(t**2) - cell_obj.coords.r * (1 / np.sqrt(1 + (a1 + 2*a2*t)**2))

    x_bot = t + - cell_obj.coords.r * ((a1 + 2 * a2 * t) / np.sqrt(1 + (a1 + 2 * a2 * t) ** 2))
    y_bot = a0 + a1*t + a2*(t**2) + cell_obj.coords.r * (1 / np.sqrt(1 + (a1 + 2*a2*t)**2))

    #Left semicirlce
    psi = np.arctan(-cell_obj.coords.p_dx(cell_obj.coords.xl))

    th_l = np.linspace(-0.5*np.pi+psi, 0.5*np.pi + psi, num=200)
    cl_dx = cell_obj.coords.r * np.cos(th_l)
    cl_dy = cell_obj.coords.r * np.sin(th_l)

    cl_x = cell_obj.coords.xl - cl_dx
    cl_y = cell_obj.coords.p(cell_obj.coords.xl) + cl_dy

    #Right semicircle
    psi = np.arctan(-cell_obj.coords.p_dx(cell_obj.coords.xr))

    th_r = np.linspace(0.5*np.pi-psi, -0.5*np.pi-psi, num=200)
    cr_dx = cell_obj.coords.r * np.cos(th_r)
    cr_dy = cell_obj.coords.r * np.sin(th_r)

    cr_x = cr_dx + cell_obj.coords.xr
    cr_y = cr_dy + cell_obj.coords.p(cell_obj.coords.xr)

    x_all = np.concatenate((cl_x[::-1], x_top, cr_x[::-1], x_bot[::-1]))
    y_all = np.concatenate((cl_y[::-1], y_top, cr_y[::-1], y_bot[::-1]))

    return x_all, y_all


register_projection(IterUpdateAxes)
register_projection(IterRedrawAxes)
