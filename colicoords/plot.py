import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import matplotlib.patches as mpatches
import seaborn.timeseries
from colicoords.config import cfg
import seaborn as sns
from scipy import stats
sns.set_style('white')



# #todo add src, python 2
# def _plot_std_bars(*args, central_data=None, ci=None, data=None, **kwargs):
#     std = data.std(axis=0)
#     ci = np.asarray((central_data - std, central_data + std))
#     kwargs.update({"central_data": central_data, "ci": ci, "data": data})
#     seaborn.timeseries._plot_ci_bars(*args, **kwargs)


# https://stackoverflow.com/questions/34293687/standard-deviation-and-errors-bars-in-seaborn-tsplot-function-in-python
def _plot_std_bars(*args, **kwargs):
    data = kwargs.pop('data')
    central_data = kwargs.pop('central_data')
    kwargs.pop('ci')

    std = data.std(axis=0)
    ci = np.asarray((central_data - std, central_data + std))
    kwargs.update({"central_data": central_data, "ci": ci, "data": data})
    seaborn.timeseries._plot_ci_bars(*args, **kwargs)


def _plot_std_band(*args, **kwargs):
    data = kwargs.pop('data')
    central_data = kwargs.pop('central_data')
    kwargs.pop('ci')

    std = data.std(axis=0)
    ci = np.asarray((central_data - std, central_data + std))
    kwargs.update({"central_data": central_data, "ci": ci, "data": data})
    seaborn.timeseries._plot_ci_band(*args, **kwargs)

seaborn.timeseries._plot_std_bars = _plot_std_bars
seaborn.timeseries._plot_std_band = _plot_std_band


class CellListPlot(object):
    def __init__(self, cell_list):
        self.cell_list = cell_list

    def hist_property(self, ax=None, tgt='length'):
        #todo update the values getting (implemented on clp)
        if tgt == 'length':
            values = np.array([c.length for c in self.cell_list]) * (cfg.IMG_PIXELSIZE / 1000)
            title = 'Cell length'
            xlabel = r'Length ($\mu m$)'
        elif tgt == 'radius':
            values = np.array([c.radius for c in self.cell_list]) * (cfg.IMG_PIXELSIZE / 1000)
            title = 'Cell radius'
            xlabel = r'Radius ($\mu m$)'
        elif tgt == 'area':
            values = np.array([c.area for c in self.cell_list]) * (cfg.IMG_PIXELSIZE / 1000)**2
            title = 'Cell area'
            xlabel = r'Area ($\mu m^{2}$)'
            #todo check these numbers!!!
        elif tgt == 'volume':
            values = np.array([c.volume for c in self.cell_list]) * (cfg.IMG_PIXELSIZE / 1000) ** 3
            title = 'Cell volume'
            xlabel = r'Volume ($\mu m^{3}$'
        else:
            raise ValueError('Invalid target')

        ax_d = sns.distplot(values, kde=False, ax=ax)
        ax_d.set_title(title)
        ax_d.set_xlabel(xlabel)
        ax_d.set_ylabel('Cell count')

        return ax_d

    def hist_intensity(self, ax=None, mask='binary', data_name='', **kwargs):
        # todo option to convert to photons?
        values = self.cell_list.get_intensity(mask=mask, data_name=data_name)

        ax_d = sns.distplot(values, kde=False, ax=ax, **kwargs)
        ax_d.set_title('Cell mean fluorescence intensity')
        ax_d.set_xlabel('Mean fluorescence (a.u.)')
        ax_d.set_ylabel('Cell count')

        return ax_d


    def plot_dist(self, ax=None, mode='r', src='', std='std_band', norm_y=False, norm_x=False, storm_weights='points', **kwargs):
        """

        :param mode: r, l, or a for radial, longitudinal or angular
        :param src: which data source to use
        :param std: band or bar style std error bars
        :param norm_y: normalize distribution wrt y
        :param norm_x normalize distribution wrt r, l, (not alpha)
        :param kwargs: are passed to plot
        :return:
        """

        if norm_x:
            stop = cfg.R_DIST_NORM_STOP
            step = cfg.R_DIST_NORM_STEP
        else:
            stop = cfg.R_DIST_STOP
            step = cfg.R_DIST_STEP

        if mode == 'r':
            x, out_arr = self.cell_list.r_dist(stop, step, data_name=src, norm_x=norm_x, storm_weight=storm_weights)
            out_arr = np.nan_to_num(out_arr)
            title = 'Radial Distribution'
        elif mode == 'l':
            raise NotImplementedError()
        elif mode == 'a':
            raise NotImplementedError()

        if norm_y:
            a_max = np.max(out_arr, axis=1)
            out_arr = out_arr / a_max[:, np.newaxis]

        t = x if norm_x else x * (cfg.IMG_PIXELSIZE / 1000)

        xunits = 'norm' if norm_x else '$\mu m$'
        yunits = 'norm' if norm_y else 'a.u.'

        ax = plt.gca() if ax is None else ax
        ax.set_xlabel('Distance ({})'.format(xunits))
        ax.set_ylabel('Intensity ({})'.format(yunits))

        if norm_y:
            ax.set_ylim(0, 1.1)

        ax_out = sns.tsplot(data=out_arr, time=t, estimator=np.mean, err_style=std, ax=ax, **kwargs)
        ax_out.set_xlabel('Distance ({})'.format(xunits))
        ax_out.set_ylabel('Signal intensity ({})'.format(yunits))
        ax_out.set_title(title)

        if norm_y:
            ax_out.set_ylim(0, 1.1)

    #def hist_intensity(self, ax=None, ):


class CellPlot(object):
    def __init__(self, cell_obj):
        self.cell_obj = cell_obj

    def plot_midline(self, ax=None, coords='mpl', **kwargs):
        """
        Plot the final found function and xl, xr
        :param coords:
        :param kwargs:
        :return:
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
        if 'interpolation' not in kwargs:
            kwargs['interpolation'] = 'nearest'

        ax = plt.gca() if ax is None else ax
        ymax, xmax = self.cell_obj.data.shape
        ax.imshow(self.cell_obj.data.binary_img, extent=[0, xmax, ymax, 0], **kwargs)

        return ax

    def plot_simulated_shape(self, ax=None, **kwargs):
        if 'interpolation' not in kwargs:
            kwargs['interpolation'] = 'nearest'
        img = self.cell_obj.coords.rc < self.cell_obj.coords.r

        ax = plt.gca() if ax is None else ax
        ymax, xmax = self.cell_obj.data.shape
        ax.imshow(img, extent=[0, xmax, ymax, 0], **kwargs)

        return ax

    def plot_bin_fit_comparison(self, ax=None, **kwargs):
        if 'interpolation' not in kwargs:
            kwargs['interpolation'] = 'nearest'
        img = self.cell_obj.coords.rc < self.cell_obj.coords.r

        ax = plt.gca() if ax is None else ax
        ymax, xmax = self.cell_obj.data.shape
        ax.imshow(3 - (2 * img + self.cell_obj.data.binary_img), extent=[0, xmax, ymax, 0], **kwargs)

        return ax
        #todo sequential colormap

    def plot_outline(self, ax=None, **kwargs):
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

    def plot_dist(self, ax=None, mode='r', src='', norm_y=False, norm_x=False, storm_weights='points'):

        if mode == 'r':
            if norm_x:
                stop = cfg.R_DIST_NORM_STOP
                step = cfg.R_DIST_NORM_STEP
            else:
                stop = cfg.R_DIST_STOP
                step = cfg.R_DIST_STEP
            x, y = self.cell_obj.r_dist(stop, step, data_name=src, norm_x=norm_x, storm_weight=storm_weights)

            if norm_y:
                y /= y.max()

        elif mode == 'l':
            raise NotImplementedError
        elif mode == 'a':
            raise NotImplementedError
        else:
            raise ValueError('Distribution mode {} not supported'.format(mode))

        x = x if norm_x else x * (cfg.IMG_PIXELSIZE / 1000)
        xunits = 'norm' if norm_x else '$\mu m$'
        yunits = 'norm' if norm_y else 'a.u.'

        ax = plt.gca() if ax is None else ax
        ax.plot(x, y)
        ax.set_xlabel('Distance ({})'.format(xunits))
        ax.set_ylabel('Intensity ({})'.format(yunits))
        if norm_y:
            ax.set_ylim(0, 1.1)

        return ax

    def plot_storm(self, data_name, ax=None, kernel=None, bw_method=0.05, upscale=2, alpha_cutoff=None, **kwargs):
        storm_table = self.cell_obj.data.data_dict[data_name]
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
        if not kernel:
            cm = plt.cm.get_cmap('Blues')
            cmap = cm if not 'cmap' in kwargs else kwargs.pop('cmap')

            img = h.T
            ax.imshow(img, interpolation='nearest', cmap=cmap, extent=[0, xmax, ymax, 0], **kwargs)
        else:
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

    def _plot_intercept_line(self, x_pos, ax=None, coords='cart', **kwargs):
        x = np.linspace(x_pos - 10, x_pos + 10, num=200)
        f_d = self.cell_obj.coords.p_dx(x_pos)
        y = (-x / f_d) + self.cell_obj.coords.p(x_pos) + (x_pos / f_d)

        #x, y = self.cell_obj.coords.transform(x, y, src='cart', tgt=coords)

        ax = plt.gca() if ax is None else ax
        ax.plot(x, y)

    def figure(self):
        plt.figure()

    def show(self):
        plt.show()
