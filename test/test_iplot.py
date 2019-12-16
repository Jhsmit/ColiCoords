from test.testcase import ArrayTestCase
import matplotlib.pyplot as plt
from colicoords.iplot import IterCellPlot, AutoIterCellPlot, iter_subplots, IterRedrawAxes, IterUpdateAxes
from colicoords.fileIO import load
import os
import numpy as np


class TestIterSubplots(ArrayTestCase):
    def test_update_axes_plot(self):
        self.fig, self.ax = iter_subplots()
        self.assertIsInstance(self.ax, IterUpdateAxes)

        i = 10
        x = [np.arange(10) for j in range(i)]
        y = [xi ** k for k, xi in enumerate(x)]

        line, = self.ax.iter_plot(x, y)
        ydata = line.get_ydata()
        self.assertArrayEqual(y[0], ydata)

        self.fig.on_next(None)
        ydata = line.get_ydata()
        self.assertArrayEqual(y[1], ydata)
        self.fig.on_prev(None)
        ydata = line.get_ydata()
        self.assertArrayEqual(y[0], ydata)
        self.fig.on_last(None)
        ydata = line.get_ydata()
        self.assertArrayEqual(y[-1], ydata)
        self.fig.on_first(None)
        ydata = line.get_ydata()
        self.assertArrayEqual(y[0], ydata)

    def test_update_axes_imshow(self):
        self.fig, self.ax = iter_subplots()

        i = 10
        imgs = [np.random.random((512, 512)) for j in range(i)]

        im = self.ax.iter_imshow(imgs)

        arr = im.get_array()
        self.assertArrayEqual(arr, imgs[0])
        self.fig.on_next(None)
        arr = im.get_array()
        self.assertArrayEqual(imgs[1], arr)
        self.fig.on_prev(None)
        arr = im.get_array()
        self.assertArrayEqual(imgs[0], arr)
        self.fig.on_last(None)
        arr = im.get_array()
        self.assertArrayEqual(imgs[-1], arr)
        self.fig.on_first(None)
        arr = im.get_array()
        self.assertArrayEqual(imgs[0], arr)

    def test_update_axes_hist(self):
        self.fig, self.ax = iter_subplots()

        i = 10
        hists = [np.random.normal(5, 2, size=200) for j in range(i)]

        n, b, p = self.ax.iter_hist(hists)

        h, edges = np.histogram(hists[0])
        hp = np.array([p.get_height() for p in self.ax.patches])
        self.assertArrayEqual(h, hp)

        self.fig.on_next(None)
        h, edges = np.histogram(hists[1])
        hp = np.array([p.get_height() for p in self.ax.patches])
        self.assertArrayEqual(h, hp)
        self.fig.on_prev(None)
        h, edges = np.histogram(hists[0])
        hp = np.array([p.get_height() for p in self.ax.patches])
        self.assertArrayEqual(h, hp)
        self.fig.on_last(None)
        h, edges = np.histogram(hists[-1])
        hp = np.array([p.get_height() for p in self.ax.patches])
        self.assertArrayEqual(h, hp)
        self.fig.on_first(None)
        h, edges = np.histogram(hists[0])
        hp = np.array([p.get_height() for p in self.ax.patches])
        self.assertArrayEqual(h, hp)

    def test_update_axes_bar(self):
        self.fig, self.ax = iter_subplots()

        i = 10
        x = [np.arange(10) for j in range(i)]
        bars = [np.random.randint(0, 100, size=10) for j in range(i)]

        bc = self.ax.iter_bar(x, bars)

        hp = np.array([p.get_height() for p in self.ax.patches])
        self.assertArrayEqual(bars[0], hp)

        self.fig.on_next(None)
        hp = np.array([p.get_height() for p in self.ax.patches])
        self.assertArrayEqual(bars[1], hp)
        self.fig.on_prev(None)
        hp = np.array([p.get_height() for p in self.ax.patches])
        self.assertArrayEqual(bars[0], hp)
        self.fig.on_last(None)
        hp = np.array([p.get_height() for p in self.ax.patches])
        self.assertArrayEqual(bars[-1], hp)
        self.fig.on_first(None)
        hp = np.array([p.get_height() for p in self.ax.patches])
        self.assertArrayEqual(bars[0], hp)

    def test_redraw_axes_plot(self):
        self.fig, self.ax = iter_subplots(subplot_kw={'projection': 'iter_redraw'})
        self.assertIsInstance(self.ax, IterRedrawAxes)

        i = 10
        x = [np.arange(10) for j in range(i)]
        y = [xi ** k for k, xi in enumerate(x)]

        line, = self.ax.iter_plot(x, y)
        ydata = line.get_ydata()
        self.assertArrayEqual(y[0], ydata)

        self.fig.on_next(None)
        line = self.ax.lines[0]
        ydata = line.get_ydata()
        self.assertArrayEqual(y[1], ydata)

        self.fig.on_prev(None)
        line = self.ax.lines[0]
        ydata = line.get_ydata()
        self.assertArrayEqual(y[0], ydata)

        self.fig.on_last(None)
        line = self.ax.lines[0]
        ydata = line.get_ydata()
        self.assertArrayEqual(y[-1], ydata)

        self.fig.on_first(None)
        line = self.ax.lines[0]
        ydata = line.get_ydata()
        self.assertArrayEqual(y[0], ydata)

    def test_redraw_axes_imshow(self):
        self.fig, self.ax = iter_subplots(subplot_kw={'projection': 'iter_redraw'})

        i = 10
        imgs = [np.random.random((512, 512)) for j in range(i)]

        im = self.ax.iter_imshow(imgs)

        arr = im.get_array()
        self.assertArrayEqual(arr, imgs[0])

        self.fig.on_next(None)
        im = self.ax.images[0]
        arr = im.get_array()
        self.assertArrayEqual(imgs[1], arr)

        self.fig.on_prev(None)
        im = self.ax.images[0]
        arr = im.get_array()
        self.assertArrayEqual(imgs[0], arr)

        self.fig.on_last(None)
        im = self.ax.images[0]
        arr = im.get_array()
        self.assertArrayEqual(imgs[-1], arr)

        self.fig.on_first(None)
        im = self.ax.images[0]
        arr = im.get_array()
        self.assertArrayEqual(imgs[0], arr)

    def test_redraw_axes_hist(self):
        self.fig, self.ax = iter_subplots(subplot_kw={'projection': 'iter_redraw'})

        i = 10
        hists = [np.random.normal(5, 2, size=200) for j in range(i)]

        n, b, p = self.ax.iter_hist(hists)

        h, edges = np.histogram(hists[0])
        hp = np.array([p.get_height() for p in self.ax.patches])
        self.assertArrayEqual(h, hp)

        self.fig.on_next(None)
        h, edges = np.histogram(hists[1])
        hp = np.array([p.get_height() for p in self.ax.patches])
        self.assertArrayEqual(h, hp)
        self.fig.on_prev(None)
        h, edges = np.histogram(hists[0])
        hp = np.array([p.get_height() for p in self.ax.patches])
        self.assertArrayEqual(h, hp)
        self.fig.on_last(None)
        h, edges = np.histogram(hists[-1])
        hp = np.array([p.get_height() for p in self.ax.patches])
        self.assertArrayEqual(h, hp)
        self.fig.on_first(None)
        h, edges = np.histogram(hists[0])
        hp = np.array([p.get_height() for p in self.ax.patches])
        self.assertArrayEqual(h, hp)

    # def test_redraw_axes_bar(self):
    #     self.fig, self.ax = iter_subplots(subplot_kw={'projection': 'iter_redraw'})
    #
    #     i = 10
    #     x = [np.arange(10) for j in range(i)]
    #     bars = [np.random.random_integers(0, 100, size=10) for j in range(i)]
    #
    #     bc = self.ax.iter_bar(x, bars)
    #
    #     hp = np.array([p.get_height() for p in self.ax.patches])
    #     self.assertArrayEqual(bars[0], hp)
    #
    #     self.fig.on_next(None)
    #     hp = np.array([p.get_height() for p in self.ax.patches])
    #     self.assertArrayEqual(bars[1], hp)
    #     self.fig.on_prev(None)
    #     hp = np.array([p.get_height() for p in self.ax.patches])
    #     self.assertArrayEqual(bars[0], hp)
    #     self.fig.on_last(None)
    #     hp = np.array([p.get_height() for p in self.ax.patches])
    #     self.assertArrayEqual(bars[-1], hp)
    #     self.fig.on_first(None)
    #     hp = np.array([p.get_height() for p in self.ax.patches])
    #     self.assertArrayEqual(bars[0], hp)


class TestIterCellPlot(ArrayTestCase):
    def setUp(self):
        f_path = os.path.dirname(os.path.realpath(__file__))
        cell_list = load(os.path.join(f_path, 'test_data', 'test_synth_cell_storm.hdf5'))
        icp = IterCellPlot(cell_list)
        # Padding
        self.cell_list = icp.cell_list

        self.icp = IterCellPlot(self.cell_list, pad=False)

        self.num = len(self.cell_list)
        self.num_st = [len(cell.data.data_dict['storm']) for cell in self.cell_list]

        self.num_poles = []
        self.num_05 = []
        for c in self.cell_list:
            st_x, st_y = c.data.data_dict['storm']['x'], c.data.data_dict['storm']['y']
            l, r, phi = c.coords.transform(st_x, st_y)
            self.num_poles.append((l == 0).sum() + (l == c.length).sum())
            self.num_05.append(((l > 0.25 * c.length) * (l < 0.75 * c.length)).sum())

        self.num_full = np.array(self.num_st) - np.array(self.num_poles)

    def test_plot_midline(self):
        fig, ax = iter_subplots()

        line = self.icp.plot_midline(ax=ax)
        x = np.linspace(self.cell_list[0].coords.xl, self.cell_list[0].coords.xr, 100)
        y = np.polyval(self.cell_list[0].coords.coeff[::-1], x)
        xl, yl = line.get_data()
        self.assertArrayAlmostEqual(x, xl, decimal=10)
        self.assertArrayAlmostEqual(y, yl, decimal=10)

        fig.on_next(None)
        x = np.linspace(self.cell_list[1].coords.xl, self.cell_list[1].coords.xr, 100)
        y = np.polyval(self.cell_list[1].coords.coeff[::-1], x)
        xl, yl = ax.lines[0].get_data()
        self.assertArrayAlmostEqual(x, xl, decimal=10)
        self.assertArrayAlmostEqual(y, yl, decimal=10)

        plt.close()

    def test_plot_binary_img(self):
        fig, ax = iter_subplots()

        image = self.icp.plot_binary_img(ax)
        data = image.get_array()
        self.assertArrayEqual(data, self.cell_list[0].data.binary_img)

        fig.on_next(None)
        image = ax.images[0]
        data = image.get_array()

        self.assertArrayEqual(data, self.cell_list[1].data.binary_img)
        plt.close()

    def test_plot_sim_binary(self):
        fig, ax = iter_subplots()

        image = self.icp.plot_simulated_binary(ax=ax)
        data = image.get_array()
        img = self.cell_list[0].coords.rc < self.cell_list[0].coords.r
        self.assertArrayEqual(data, img)

        fig.on_next(None)
        image = ax.images[0]
        data = image.get_array()
        img = self.cell_list[1].coords.rc < self.cell_list[1].coords.r

        self.assertArrayEqual(data, img)
        plt.close()

    def test_plot_bin_fit_comparison(self):
        fig, ax = iter_subplots()

        image = self.icp.plot_bin_fit_comparison(ax=ax)
        data = image.get_array()
        img = self.cell_list[0].coords.rc < self.cell_list[0].coords.r
        final_img = 3 - (2 * img + self.cell_list[0].data.binary_img)
        self.assertArrayEqual(data, final_img)

        fig.on_next(None)
        data = image.get_array()
        img = self.cell_list[1].coords.rc < self.cell_list[1].coords.r
        final_img = 3 - (2 * img + self.cell_list[1].data.binary_img)
        self.assertArrayEqual(data, final_img)
        plt.close()

    def test_plot_outline(self):
        fig, ax = iter_subplots()

        line, = self.icp.plot_outline(ax=ax)
        x, y = line.get_data()

        dist = np.sqrt((x[:-1] - x[1:])**2 + (y[:-1] - y[1:])**2)
        self.assertTrue(np.all(dist < 0.14))
        self.assertEqual(0.10525317206276026, np.mean(dist))

        fig.on_next(None)
        fig.on_prev(None)

        x1, y1 = ax.lines[0].get_data()
        self.assertArrayEqual(x1, x)
        self.assertArrayEqual(y1, y)

        plt.close()

    def test_plot_r_dist(self):
        fig, ax = iter_subplots()

        line = self.icp.plot_r_dist(ax=ax)
        x, y = line.get_data()

        fig.on_next(None)
        fig.on_prev(None)

        x1, y1 = ax.lines[0].get_data()
        self.assertArrayEqual(x1, x)
        self.assertArrayEqual(y1, y)

        fig.on_random(None)
        fig.on_first(None)

        x2, y2 = ax.lines[0].get_data()
        self.assertArrayEqual(x2, x)
        self.assertArrayEqual(y2, y)

        plt.close()

    def test_plot_l_dist(self):
        fig, ax = iter_subplots()

        line = self.icp.plot_l_dist(ax=ax)
        x, y = line.get_data()

        fig.on_next(None)
        fig.on_prev(None)

        x1, y1 = ax.lines[0].get_data()
        self.assertArrayEqual(x1, x)
        self.assertArrayEqual(y1, y)

        fig.on_random(None)
        fig.on_first(None)

        x2, y2 = ax.lines[0].get_data()
        self.assertArrayEqual(x2, x)
        self.assertArrayEqual(y2, y)

    def test_plot_phi_dist(self):
        fig, ax = iter_subplots()

        line_l, line_r = self.icp.plot_phi_dist(ax=ax)
        x_l, y_l = line_l.get_data()
        x_r, y_r = line_r.get_data()

        fig.on_next(None)
        fig.on_prev(None)

        x1_l, y1_l = ax.lines[0].get_data()
        self.assertArrayEqual(x1_l, x_l)
        self.assertArrayEqual(y1_l, y_l)

        xr_1, yr_1 = ax.lines[1].get_data()
        self.assertArrayEqual(xr_1, x_r)
        self.assertArrayEqual(yr_1, y_r)

        fig.on_random(None)
        fig.on_first(None)

        x1_l, y1_l = ax.lines[0].get_data()
        self.assertArrayEqual(x1_l, x_l)
        self.assertArrayEqual(y1_l, y_l)

        xr_1, yr_1 = ax.lines[1].get_data()
        self.assertArrayEqual(xr_1, x_r)
        self.assertArrayEqual(yr_1, y_r)

    def test_plot_storm(self):
        fig, ax = iter_subplots()
        img = self.icp.plot_storm(ax=ax, method='hist')
        data = img.get_array()

        num = len(self.cell_list[0].data.data_dict['storm'])
        self.assertEqual(data.sum(), num)

        fig.on_next(None)
        data = ax.images[0].get_array()
        num = len(self.cell_list[1].data.data_dict['storm'])
        self.assertEqual(data.sum(), num)
        plt.close()

    def test_plot_l_class(self):
        fig, ax = iter_subplots()
        container = self.icp.plot_l_class(ax=ax)
        self.assertEqual(len(container), 3)

        h = [rect.get_height() for rect in container]
        num = len(self.cell_list[0].data.data_dict['storm'])
        self.assertEqual(num, sum(h))
        self.assertEqual(h[0], self.num_poles[0])

        fig.on_next(None)
        h = [rect.get_height() for rect in ax.patches]
        num = len(self.cell_list[1].data.data_dict['storm'])
        self.assertEqual(num, sum(h))
        self.assertEqual(h[0], self.num_poles[1])
        plt.close()

    def test_hist_l_storm(self):
        fig, ax = iter_subplots()
        n, b, p = self.icp.hist_l_storm(ax=ax)
        self.assertEqual(self.num_st[0], np.sum(n))

        fig.on_next(None)
        num = np.sum([p.get_height() for p in ax.patches])
        self.assertEqual(self.num_st[1], num)
        plt.close()

    def test_hist_r_storm(self):
        fig, ax = iter_subplots()
        n, b, p = self.icp.hist_r_storm(ax=ax)
        self.assertEqual(self.num_st[0], np.sum(n))

        fig.on_next(None)
        num = np.sum([p.get_height() for p in ax.patches])
        self.assertEqual(self.num_st[1], num)
        plt.close()

    def test_hist_phi_storm(self):
        fig, ax = iter_subplots()
        n, b, p = self.icp.hist_phi_storm(ax=ax)
        self.assertEqual(self.num_poles[0], np.sum(n))

        fig.on_next(None)
        num = np.sum([p.get_height() for p in ax.patches])
        self.assertEqual(self.num_poles[1], num)

        plt.close()

    def test_misc(self):
        fig = plt.figure()
        self.icp.savefig('deleteme.png')
        self.icp.show()
        plt.close()