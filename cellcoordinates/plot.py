import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches


class CellPlot(object):
    def __init__(self, cell_obj):
        self.c = cell_obj

    def plot_final_func(self, coords='mpl', **kwargs):
        """
        Plot the final found function and xl, xr
        :param coords:
        :param kwargs:
        :return:
        """
        x = np.linspace(self.c.xl, self.c.xr, 100)
        y = self.c.coords.p(x)
        if 'color' not in kwargs:
            kwargs['color'] = 'r'
        if coords == 'mpl':
            x, y = self.c.coords.transform(x, y, src='cart', tgt='mpl')
        plt.plot(x, y, **kwargs)

    def plot_binary_img(self, **kwargs):
        if 'interpolation' not in kwargs:
            kwargs['interpolation'] = 'nearest'
        plt.imshow(self.c.data.binary_img, **kwargs)

    def plot_simulated_shape(self, **kwargs):
        if 'interpolation' not in kwargs:
            kwargs['interpolation'] = 'nearest'
        img = self.c.coords.rc < self.c.coords.r
        plt.imshow(img, **kwargs)

    def plot_bin_fit_comparison(self, **kwargs):
        if 'interpolation' not in kwargs:
            kwargs['interpolation'] = 'nearest'
        img = self.c.coords.rc < self.c.coords.r
        plt.imshow(3 - (2*img + self.c.data.binary_img), **kwargs)
        #todo sequential colormap

    def plot_outline(self, coords='cart', **kwargs):
        #todo: works but: semicircles are not exactly from 0 to 180 but instead depend on local slope (xr, xl)
        #todo: dx sign depends on slope sign (f_d > 0, dx < 0), vice versa?

        x = np.linspace(self.c.coords.xl, self.c.coords.xr, 500)
        p_dx = self.c.coords.p_dx(x)

        dy_t = np.sqrt(self.c.coords.r**2 * (1 + 1 / (1 + (1 / p_dx**2))))
        dx_t = np.sqrt(self.c.coords.r**2 / (1 + (1 / p_dx**2)))
        x_t = x - ((p_dx/np.abs(p_dx)) * dx_t)
        y_t = self.c.coords.p(x) + dy_t

        x_b = (x + ((p_dx/np.abs(p_dx)) * dx_t))[::-1]
        y_b = (self.c.coords.p(x) - dy_t)[::-1]

        #Left semicirlce
        psi = np.arctan(-self.c.coords.p_dx(self.c.coords.xl))

        th_l = np.linspace(-0.5*np.pi+psi, 0.5*np.pi + psi, num=200)
        cl_dx = self.c.coords.r*np.cos(th_l)
        cl_dy = self.c.coords.r*np.sin(th_l)

        cl_x = self.c.coords.xl - cl_dx
        cl_y = self.c.coords.p(self.c.coords.xl) + cl_dy

        #Right semicircle
        psi = np.arctan(-self.c.coords.p_dx(self.c.coords.xr))

        th_r = np.linspace(0.5*np.pi-psi, -0.5*np.pi-psi, num=200)
        cr_dx = self.c.coords.r*np.cos(th_r)
        cr_dy = self.c.coords.r*np.sin(th_r)

        cr_x = cr_dx + self.c.coords.xr
        cr_y = cr_dy + self.c.coords.p(self.c.coords.xr)

        x_all = np.concatenate((cl_x, x_t, cr_x, x_b))
        y_all = np.concatenate((cl_y, y_t, cr_y, y_b))

        x_all, y_all = self.c.coords.transform(x_all, y_all, src='cart', tgt=coords)

        plt.plot(x_all, y_all, color='r', **kwargs)

    def _plot_intercept_line(self, x_pos, coords='cart', **kwargs):
        x = np.linspace(x_pos - 10, x_pos + 10, num=200)
        f_d = self.c.coords.p_dx(x_pos)
        y = (-x / f_d) + self.c.coords.p(x_pos) + (x_pos / f_d)

        x, y = self.c.coords.transform(x, y, src='cart', tgt=coords)

        plt.plot(x, y)

    def show(self):
        plt.show()