from test.testcase import ArrayTestCase
import matplotlib.pyplot as plt
from colicoords.data_models import Data
from colicoords.plot import CellPlot
from colicoords.fileIO import load
import os
import numpy as np
import unittest


#todo include storm intensity field
class TestCellPlot(ArrayTestCase):
    def setUp(self):
        f_path = os.path.dirname(os.path.realpath(__file__))
        self.cell_list = load(os.path.join(f_path, 'test_data', 'test_synth_cell_storm.hdf5'))
        self.cell = self.cell_list[0]
        self.cp = CellPlot(self.cell)

        x = np.arange(20)
        y = np.exp(-x / 5)

        img_3d = self.cell.data.data_dict['fluorescence'][np.newaxis, :, :] * y[:, np.newaxis, np.newaxis]
        self.cell.data.add_data(img_3d, 'fluorescence', 'flu_3d')

    def test_plot_midline(self):
        fig, ax = plt.subplots()

        line = self.cp.plot_midline(ax=ax)

        x = np.linspace(self.cell.coords.xl, self.cell.coords.xr, 100)
        y = np.polyval(self.cell.coords.coeff[::-1], x)
        xl, yl = line.get_data()
        self.assertArrayEqual(y, yl)

        line = self.cp.plot_midline(color='g')
        fig.close()

    def test_plot_binary_img(self):
        fig, ax = plt.subplots()

        image = self.cp.plot_binary_img(ax)
        data = image.get_array()
        self.assertArrayEqual(data, self.cell.data.binary_img())

        fig.close()

    def test_plot_sim_binary(self):
        fig, ax = plt.subplots()

        image = self.cp.plot_simulated_binary(ax=ax)
        data = image.get_array()
        img = self.cell.coords.rc < self.cell.coords.r
        self.assertArrayEqual(data, img)
        fig.close()

    def test_plot_bin_fit_comparison(self):
        fig, ax = plt.subplots()

        image = self.cp.plot_bin_fit_comparison(ax=ax)
        data = image.get_array()
        img = self.cell.coords.rc < self.cell.coords.r
        final_img = 3 - (2 * img + self.cell.data.binary_img)
        self.assertArrayEqual(data, final_img)
        fig.close()

    def test_plot_outline(self):
        fig, ax = plt.subplots()

        line = self.cp.plot_outline(ax=ax)
        x, y = line.get_data()

        dist = np.sqrt((x[:-1] - x[1:])**2 + (y[:-1] - y[1:])**2)
        self.assertTrue(np.all(dist < 0.14))
        self.assertEqual(0.10525317206276026, np.mean(dist))
        fig.close()

    def test_plot_r_dist(self):
        fig, ax = plt.subplots()

        line = self.cp.plot_r_dist(ax=ax)
        x1, y1 = line.get_data()

        line = self.cp.plot_r_dist(ax=ax, data_name='fluorescence')
        x2, y2 = line.get_data()
        self.assertArrayEqual(x1, x2)
        self.assertArrayEqual(y1, y2)
        self.assertEqual(y2[0], 0.04789108896255459)

        line = self.cp.plot_r_dist(ax=ax, zero=True)
        x, y = line.get_data()
        self.assertEqual(0, y.min())
        label = ax.get_ylabel()
        self.assertEqual(label, 'Intensity (a.u.)')

        line = self.cp.plot_r_dist(ax=ax, norm_y=True)
        x, y = line.get_data()
        self.assertEqual(1, y.max())
        label = ax.get_ylabel()
        self.assertEqual(label, 'Intensity (norm)')

        line = self.cp.plot_r_dist(ax=ax, zero=True, norm_y=True)
        x, y = line.get_data()
        self.assertEqual(0, y.min())
        self.assertEqual(1, y.max())
        label = ax.get_ylabel()
        self.assertEqual(label, 'Intensity (norm)')

        line = self.cp.plot_r_dist(ax=ax, norm_x=True)
        label = ax.get_xlabel()
        self.assertEqual(label, 'Distance (norm)')
        line = self.cp.plot_r_dist(ax=ax, method='box')
        x, y = line.get_data()
        self.assertEqual(y[0], 0.04788923286631714)

        ax.clear()

        line = self.cp.plot_r_dist(data_name='storm', method='box')
        x, y = line.get_data()
        self.assertEqual(y.sum(), len(self.cell.data.data_dict['storm']))

        st_x, st_y = self.cell.data.data_dict['storm']['x'], self.cell.data.data_dict['storm']['y']
        l, r, phi = self.cell.coords.transform(st_x, st_y)

        line = self.cp.plot_r_dist(data_name='storm', method='box', limit_l='poles')
        x, y = line.get_data()
        num = (l == 0).sum() + (l == self.cell.length).sum()
        self.assertEqual(y.sum(), num)

        line = self.cp.plot_r_dist(data_name='storm', method='box', limit_l='full')
        x, y = line.get_data()
        num = len(st_x) - num
        self.assertEqual(y.sum(), num)

        line = self.cp.plot_r_dist(data_name='storm', method='box', limit_l=0.5)
        x, y = line.get_data()
        num = np.sum((l > 0.25*self.cell.length)*(l < 0.75*self.cell.length))
        self.assertEqual(y.sum(), num)
        fig.close()

    def test_plot_l_dist(self):
        fig, ax = plt.subplots()

        line = self.cp.plot_l_dist(ax=ax)
        x1, y1 = line.get_data()

        line = self.cp.plot_l_dist(ax=ax, data_name='fluorescence')
        x2, y2 = line.get_data()
        self.assertArrayEqual(x1, x2)
        self.assertArrayEqual(y1, y2)
        self.assertEqual(y2[0], 0.039903571495215485)

        line = self.cp.plot_l_dist(ax=ax, zero=True)
        x, y = line.get_data()
        self.assertEqual(0, y.min())
        label = ax.get_ylabel()
        self.assertEqual(label, 'Intensity (a.u.)')

        line = self.cp.plot_l_dist(ax=ax, norm_y=True)
        x, y = line.get_data()
        self.assertEqual(1, y.max())
        label = ax.get_ylabel()
        self.assertEqual(label, 'Intensity (norm)')

        line = self.cp.plot_l_dist(ax=ax, zero=True, norm_y=True)
        x, y = line.get_data()
        self.assertEqual(0, y.min())
        self.assertEqual(1, y.max())
        label = ax.get_ylabel()
        self.assertEqual(label, 'Intensity (norm)')

        line = self.cp.plot_l_dist(ax=ax, norm_x=True)
        label = ax.get_xlabel()
        self.assertEqual(label, 'Distance (norm)')
        line = self.cp.plot_l_dist(ax=ax, method='box')
        x, y = line.get_data()
        self.assertEqual(y[0], 0.0)

        ax.clear()

        line = self.cp.plot_l_dist(data_name='storm', method='box', r_max=np.inf)
        x, y = line.get_data()
        self.assertEqual(y.sum(), len(self.cell.data.data_dict['storm']))

        st_x, st_y = self.cell.data.data_dict['storm']['x'], self.cell.data.data_dict['storm']['y']
        l, r, phi = self.cell.coords.transform(st_x, st_y)

        line = self.cp.plot_l_dist(data_name='storm', method='box')
        x, y = line.get_data()
        num = (r < self.cell.coords.r).sum()
        self.assertEqual(y.sum(), num)
        fig.close()

    def test_plot_phi_dist(self):
        fig, ax = plt.subplots()

        line_l, line_r = self.cp.plot_phi_dist(ax=ax)
        x1_l, y1_l = line_l.get_data()
        x1_r, y1_r = line_r.get_data()

        line_l, line_r = self.cp.plot_phi_dist(ax=ax, data_name='fluorescence')
        x2_l, y2_l = line_l.get_data()
        x2_r, y2_r = line_r.get_data()
        self.assertArrayEqual(x1_l, x2_l)
        self.assertArrayEqual(y1_l, y2_l)

        self.assertArrayEqual(x1_r, x2_r)
        self.assertArrayEqual(y1_r, y2_r)
        print(y1_r[0], y1_r[0])
        #self.assertEqual(y2[0], 0.039903571495215485)

        line_l, line_r = self.cp.plot_phi_dist(ax=ax, data_name='fluorescence', r_min=2)
        line_l, line_r = self.cp.plot_phi_dist(ax=ax, data_name='fluorescence', r_max=0)

        x, y = line_l.get_data()
        self.assertTrue(np.all(y == 0))

        x, y = line_r.get_data()
        self.assertTrue(np.all(y == 0))

        line_l, line_r = self.cp.plot_phi_dist(ax=ax, method='box')
        x, y = line_l.get_data()
        self.assertEqual(y[0], 0.0)

        x, y = line_r.get_data()

        self.assertEqual(y[0], 0.0)

        ax.clear()

        line = self.cp.plot_l_dist(data_name='storm', method='box', r_max=np.inf)
        x, y = line.get_data()
        self.assertEqual(y.sum(), len(self.cell.data.data_dict['storm']))

        st_x, st_y = self.cell.data.data_dict['storm']['x'], self.cell.data.data_dict['storm']['y']
        l, r, phi = self.cell.coords.transform(st_x, st_y)

        line = self.cp.plot_l_dist(data_name='storm', method='box')
        x, y = line.get_data()
        num = (r < self.cell.coords.r).sum()
        self.assertEqual(y.sum(), num)
        fig.close()

    def test_plot_storm(self):
        st_x, st_y = self.cell.data.data_dict['storm']['x'], self.cell.data.data_dict['storm']['y']
        num = len(st_x)

        fig, ax = plt.subplots()
        img = self.cp.plot_storm(ax=ax, method='hist')
        data = img.get_array()
        self.assertEqual(data.sum(), num)

        img = self.cp.plot_storm(ax=ax, method='gauss', upscale=2)
        data = img.get_array()
        self.assertEqual(data.sum(), 13638.106321000001)
        self.assertEqual(data.ndim, 3)
        fig.close()

    def test_plot_l_class(self):
        st_x, st_y = self.cell.data.data_dict['storm']['x'], self.cell.data.data_dict['storm']['y']
        l, r, phi = self.cell.coords.transform(st_x, st_y)
        num = len(st_x)

        fig, ax = plt.subplots()
        container = self.cp.plot_l_class(ax=ax)
        self.assertEqual(len(container), 3)

        h = [rect.get_height() for rect in container]
        self.assertEqual(num, sum(h))

        num = (l == 0).sum() + (l == self.cell.length).sum()
        self.assertEqual(num, h[0])
        fig.close()

    def test_plot_kymograph(self):
        fig, ax = plt.subplots()
        self.cp.plot_kymograph(ax=ax, data_name='flu_3d', norm_y=False)
        # fig.close() cant close this figure?

        with self.assertRaises(NotImplementedError):
            self.cp.plot_kymograph(data_name='flu_3d', mode='l')
        with self.assertRaises(NotImplementedError):
            self.cp.plot_kymograph(data_name='flu_3d', mode='a')

    def test_hist_l_storm(self):
        st_x, st_y = self.cell.data.data_dict['storm']['x'], self.cell.data.data_dict['storm']['y']
        l, r, phi = self.cell.coords.transform(st_x, st_y)
        num = len(st_x)

        fig, ax = plt.subplots()
        n, b, p = self.cp.hist_l_storm(ax=ax)
        self.assertEqual(num, np.sum(n))

        n, b, p = self.cp.hist_l_storm(ax=ax, norm_x=True)
        self.assertEqual(num, np.sum(n))
        fig.close()

    def test_hist_r_storm(self):
        st_x, st_y = self.cell.data.data_dict['storm']['x'], self.cell.data.data_dict['storm']['y']
        l, r, phi = self.cell.coords.transform(st_x, st_y)
        num = len(st_x)

        fig, ax = plt.subplots()
        n, b, p = self.cp.hist_r_storm(ax=ax)
        self.assertEqual(num, np.sum(n))

        n, b, p = self.cp.hist_r_storm(ax=ax, norm_x=True)
        self.assertEqual(num, np.sum(n))

    def test_hist_phi_storm(self):
        st_x, st_y = self.cell.data.data_dict['storm']['x'], self.cell.data.data_dict['storm']['y']
        l, r, phi = self.cell.coords.transform(st_x, st_y)
        num = len(st_x)

        fig, ax = plt.subplots()
        n, b, p = self.cp.hist_phi_storm(ax=ax)
        self.assertEqual(num, np.sum(n))

        n, b, p = self.cp.hist_phi_storm(ax=ax, norm_x=True)
        self.assertEqual(num, np.sum(n))

    def test_misc(self):

        img = np.random.rand(*self.cell.data.binary_img.shape)
        fig = self.cp.figure()
        self.cp.imshow(img)
        self.cp.savefig('deleteme.png')
