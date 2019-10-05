from test.testcase import ArrayTestCase
import matplotlib.pyplot as plt
from colicoords.plot import CellPlot, CellListPlot
from colicoords.fileIO import load
import os
import numpy as np


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
        self.assertArrayAlmostEqual(y, yl, decimal=10)

        line = self.cp.plot_midline(color='g')
        plt.close()

    def test_plot_binary_img(self):
        fig, ax = plt.subplots()

        image = self.cp.plot_binary_img(ax)
        data = image.get_array()
        self.assertArrayEqual(data, self.cell.data.binary_img)

        plt.close()

    def test_plot_sim_binary(self):
        fig, ax = plt.subplots()

        image = self.cp.plot_simulated_binary(ax=ax)
        data = image.get_array()
        img = self.cell.coords.rc < self.cell.coords.r
        self.assertArrayEqual(data, img)
        plt.close()

    def test_plot_bin_fit_comparison(self):
        fig, ax = plt.subplots()

        image = self.cp.plot_bin_fit_comparison(ax=ax)
        data = image.get_array()
        img = self.cell.coords.rc < self.cell.coords.r
        final_img = 3 - (2 * img + self.cell.data.binary_img)
        self.assertArrayEqual(data, final_img)
        plt.close()

    def test_plot_outline(self):
        fig, ax = plt.subplots()

        line = self.cp.plot_outline(ax=ax)
        x, y = line.get_data()

        dist = np.sqrt((x[:-1] - x[1:])**2 + (y[:-1] - y[1:])**2)
        self.assertTrue(np.all(dist < 0.14))
        self.assertEqual(0.10525317206276026, np.mean(dist))
        plt.close()

    def test_plot_r_dist(self):
        fig, ax = plt.subplots()

        line = self.cp.plot_r_dist(ax=ax)
        x1, y1 = line.get_data()

        line = self.cp.plot_r_dist(ax=ax, data_name='fluorescence')
        x2, y2 = line.get_data()
        self.assertArrayEqual(x1, x2)
        self.assertArrayEqual(y1, y2)
        self.assertAlmostEqual(y2[0], 0.04789108896255459, 10)

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
        self.assertAlmostEqual(y[0], 0.04788923286631714, 10)

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
        plt.close()

    def test_plot_l_dist(self):
        fig, ax = plt.subplots()

        line = self.cp.plot_l_dist(ax=ax)
        x1, y1 = line.get_data()

        line = self.cp.plot_l_dist(ax=ax, data_name='fluorescence')
        x2, y2 = line.get_data()
        self.assertArrayEqual(x1, x2)
        self.assertArrayEqual(y1, y2)
        self.assertAlmostEqual(y2[0], 0.039903571495215485, 10)

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
        plt.close()

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
        self.assertAlmostEqual(y1_l[0], 0.05548356357544916, 10)
        self.assertAlmostEqual(y1_r[0], 0.054010840527480314, 10)

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

        line_l, line_r = self.cp.plot_phi_dist(data_name='storm', method='box', r_max=np.inf)
        x, yl = line_l.get_data()
        x, yr = line_r.get_data()

        st_x, st_y = self.cell.data.data_dict['storm']['x'], self.cell.data.data_dict['storm']['y']
        l, r, phi = self.cell.coords.transform(st_x, st_y)
        num = (l == 0).sum() + (l == self.cell.length).sum()
        self.assertEqual(yl.sum() + yr.sum(), num)

        line_l, line_r = self.cp.plot_phi_dist(data_name='storm', method='box')
        x, yl = line_l.get_data()
        x, yr = line_r.get_data()
        num = ( (r < self.cell.coords.r) * np.logical_or(l == 0, l == self.cell.length) ).sum()
        self.assertEqual(yl.sum() + yr.sum(), num)
        plt.close()

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
        plt.close()

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
        plt.close()

    def test_plot_kymograph(self):
        fig, ax = plt.subplots()
        self.cp.plot_kymograph(ax=ax, data_name='flu_3d', norm_y=False)
        # plt.close() cant close this figure?

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
        plt.close()

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
        num = (l == 0).sum() + (l == self.cell.length).sum()

        fig, ax = plt.subplots()
        n, b, p = self.cp.hist_phi_storm(ax=ax)
        self.assertEqual(num, np.sum(n))
        plt.close()

    def test_misc(self):

        img = np.random.rand(*self.cell.data.binary_img.shape)
        fig = self.cp.figure()
        self.cp.imshow(img)
        self.cp.savefig('deleteme.png')
        plt.close()


class TestCellListPlot(ArrayTestCase):
    def setUp(self):
        f_path = os.path.dirname(os.path.realpath(__file__))
        self.cell_list = load(os.path.join(f_path, 'test_data', 'test_synth_cell_storm.hdf5'))
        self.num = len(self.cell_list)
        self.num_st = np.sum([len(cell.data.data_dict['storm']) for cell in self.cell_list])
        self.clp = CellListPlot(self.cell_list)

        self.num_poles = 0
        self.num_05 = 0
        for c in self.cell_list:
            st_x, st_y = c.data.data_dict['storm']['x'], c.data.data_dict['storm']['y']
            l, r, phi = c.coords.transform(st_x, st_y)
            self.num_poles += ((l == 0).sum() + (l == c.length).sum() )
            self.num_05 += np.sum((l > 0.25 * c.length) * (l < 0.75 * c.length))
        self.num_full = self.num_st - self.num_poles

    def test_hist_property(self):
        fig, ax = plt.subplots()
        for p in ['length', 'radius', 'area', 'surface', 'volume']:
            n, b, p = self.clp.hist_property(prop=p, ax=ax)
            self.assertEqual(self.num, np.sum(n))
            self.assertEqual(ax.get_ylabel(), 'Cell count')
        ax.clear()

        with self.assertRaises(ValueError):
            self.clp.hist_property(prop='asdf')

        n, b, p = self.clp.hist_property(prop='radius', ax=ax, bins=[0, 1000])
        self.assertEqual(len(b), 2)
        plt.close()

    def test_hist_intensity(self):
        fig, ax = plt.subplots()

        n, b, p = self.clp.hist_intensity(ax=ax)
        self.assertEqual(self.num, np.sum(n))

        n, b, p = self.clp.hist_intensity(ax=ax, mask='coords')
        self.assertEqual(self.num, np.sum(n))

    def test_plot_r_dist(self):
        fig, ax = plt.subplots()

        line = self.clp.plot_r_dist(ax=ax)
        x1, y1 = line.get_data()

        line = self.clp.plot_r_dist(ax=ax, data_name='fluorescence')
        x2, y2 = line.get_data()
        self.assertArrayEqual(x1, x2)
        self.assertArrayEqual(y1, y2)
        self.assertAlmostEqual(y2[0], 0.05025988419589037, 10)

        line = self.clp.plot_r_dist(ax=ax, zero=True)
        x, y = line.get_data()
        self.assertEqual(0, y.min())
        label = ax.get_ylabel()
        self.assertEqual(label, 'Intensity (a.u.)')

        line = self.clp.plot_r_dist(ax=ax, norm_y=True)
        x, y = line.get_data()
        # This != 1 because curves are individually normalized and then averaged
        self.assertEqual(0.9891755849204928, y.max())
        label = ax.get_ylabel()
        self.assertEqual(label, 'Intensity (norm)')

        line = self.clp.plot_r_dist(ax=ax, zero=True, norm_y=True)
        x, y = line.get_data()
        self.assertEqual(0, y.min())
        self.assertEqual(0.9891755849204829, y.max())
        label = ax.get_ylabel()
        self.assertEqual(label, 'Intensity (norm)')

        line = self.clp.plot_r_dist(ax=ax, norm_x=True)
        label = ax.get_xlabel()
        self.assertEqual(label, 'Distance (norm)')

        line = self.clp.plot_r_dist(ax=ax, method='box')
        x, y = line.get_data()
        self.assertAlmostEqual(y[0], 0.05018975533708976, 10)

        ax.clear()

        line = self.clp.plot_r_dist(data_name='storm', method='box')
        x, y = line.get_data()
        self.assertEqual(y.sum(), self.num_st / self.num)

        line = self.clp.plot_r_dist(data_name='storm', method='box', limit_l='poles')
        x, y = line.get_data()
        self.assertAlmostEqual(y.sum(), self.num_poles / self.num, 10)

        line = self.clp.plot_r_dist(data_name='storm', method='box', limit_l='full')
        x, y = line.get_data()
        self.assertAlmostEqual(y.sum(), self.num_full / self.num, 10)

        line = self.clp.plot_r_dist(data_name='storm', method='box', limit_l=0.5)
        x, y = line.get_data()
        self.assertAlmostEqual(y.sum(), self.num_05 / self.num, 10)
        plt.close()

    def test_plot_l_dist(self):
        fig, ax = plt.subplots()

        line = self.clp.plot_l_dist(ax=ax)
        x1, y1 = line.get_data()

        line = self.clp.plot_l_dist(ax=ax, data_name='fluorescence')
        x2, y2 = line.get_data()
        self.assertArrayEqual(x1, x2)
        self.assertArrayEqual(y1, y2)
        self.assertAlmostEqual(y2[0], 0.04317437431747161, 10)

        line = self.clp.plot_l_dist(ax=ax, zero=True)
        x, y = line.get_data()
        # != 0 because lines are individually normalized, zero point differs along x
        self.assertAlmostEqual(0.0008804900221751369, y.min(), 10)
        label = ax.get_ylabel()
        self.assertEqual(label, 'Intensity (a.u.)')

        line = self.clp.plot_l_dist(ax=ax, norm_y=True)
        x, y = line.get_data()
        self.assertAlmostEqual(0.989601334046804, y.max(), 10)
        label = ax.get_ylabel()
        self.assertEqual(label, 'Intensity (norm)')

        line = self.clp.plot_l_dist(ax=ax, zero=True, norm_y=True)
        x, y = line.get_data()
        self.assertAlmostEqual(0.0414570895171721, y.min(), 10)
        self.assertAlmostEqual(0.9693266707814306, y.max(), 10)
        label = ax.get_ylabel()
        self.assertEqual(label, 'Intensity (norm)')

        line = self.clp.plot_l_dist(ax=ax, method='box')
        x, y = line.get_data()
        self.assertEqual(y[0], 0.0)

        ax.clear()

        line = self.clp.plot_l_dist(data_name='storm', method='box', r_max=np.inf)
        x, y = line.get_data()
        self.assertAlmostEqual(y.sum(), self.num_st / self.num, 0)  # 350.2 vs 350.25

        r_num = 0
        for c in self.cell_list:
            st_x, st_y = c.data.data_dict['storm']['x'], c.data.data_dict['storm']['y']
            l, r, phi = c.coords.transform(st_x, st_y)
            r_num += (r < c.coords.r).sum()

        line = self.clp.plot_l_dist(data_name='storm', method='box')
        x, y = line.get_data()
        self.assertEqual(y.sum(), r_num / self.num)
        plt.close()

    def test_plot_phi_dist(self):
        fig, ax = plt.subplots()

        line_l, line_r = self.clp.plot_phi_dist(ax=ax)
        x1_l, y1_l = line_l.get_data()
        x1_r, y1_r = line_r.get_data()

        line_l, line_r = self.clp.plot_phi_dist(ax=ax, data_name='fluorescence')
        x2_l, y2_l = line_l.get_data()
        x2_r, y2_r = line_r.get_data()
        self.assertArrayEqual(x1_l, x2_l)
        self.assertArrayEqual(y1_l, y2_l)

        self.assertArrayEqual(x1_r, x2_r)
        self.assertArrayEqual(y1_r, y2_r)
        self.assertAlmostEqual(y1_l[0], 0.05852775222222941, 10)
        self.assertAlmostEqual(y1_r[0], 0.05864176599075832, 10)

        line_l, line_r = self.clp.plot_phi_dist(ax=ax, data_name='fluorescence', r_min=2)
        line_l, line_r = self.clp.plot_phi_dist(ax=ax, data_name='fluorescence', r_max=0)

        x, y = line_l.get_data()
        self.assertTrue(np.all(y == 0))

        x, y = line_r.get_data()
        self.assertTrue(np.all(y == 0))

        line_l, line_r = self.clp.plot_phi_dist(ax=ax, method='box')
        x, y = line_l.get_data()
        self.assertAlmostEqual(y[0], 0.020184222827773407, 10)

        x, y = line_r.get_data()

        self.assertAlmostEqual(y[0], 0.017214633694777406, 10)

        ax.clear()

        line_l, line_r = self.clp.plot_phi_dist(data_name='storm', method='box', r_max=np.inf)
        x, yl = line_l.get_data()
        x, yr = line_r.get_data()

        self.assertAlmostEqual(yl.sum() + yr.sum(), self.num_poles / self.num, 1)

        line_l, line_r = self.clp.plot_phi_dist(data_name='storm', method='box')
        x, yl = line_l.get_data()
        x, yr = line_r.get_data()
        num_poles_r = 0
        for c in self.cell_list:
            st_x, st_y = c.data.data_dict['storm']['x'], c.data.data_dict['storm']['y']
            l, r, phi = c.coords.transform(st_x, st_y)
            num_poles_r += ( (r < c.coords.r) * np.logical_or(l == 0, l == c.length) ).sum()

        self.assertAlmostEqual(float(yl.sum() + yr.sum()),  float(num_poles_r / self.num), 1)

    def test_plot_class(self):
        fig, ax = plt.subplots()
        container = self.clp.plot_l_class(ax=ax)
        self.assertEqual(len(container), 3)

        h = [rect.get_height() for rect in container]
        self.assertEqual(self.num_st / len(self.cell_list), sum(h))

        num = 0
        for c in self.cell_list:
            st_x, st_y = c.data.data_dict['storm']['x'], c.data.data_dict['storm']['y']
            l, r, phi = c.coords.transform(st_x, st_y)
            num += (l == 0).sum() + (l == c.length).sum()

        self.assertEqual(self.num_poles / len(self.cell_list), h[0])
        plt.close()

    def test_plot_kymograph(self):
        fig, ax = plt.subplots()
        self.clp.plot_kymograph(ax=ax, norm_y=False)
        # plt.close() cant close this figure?

        with self.assertRaises(NotImplementedError):
            self.clp.plot_kymograph(data_name='flu_3d', mode='l')
        with self.assertRaises(NotImplementedError):
            self.clp.plot_kymograph(data_name='flu_3d', mode='a')

    def test_hist_l_storm(self):
        fig, ax = plt.subplots()
        n, b, p = self.clp.hist_l_storm(ax=ax)
        self.assertEqual(self.num_st, np.sum(n))

        n, b, p = self.clp.hist_l_storm(ax=ax)
        self.assertEqual(self.num_st, np.sum(n))
        plt.close()

    def test_hist_r_storm(self):
        fig, ax = plt.subplots()
        n, b, p = self.clp.hist_r_storm(ax=ax)
        self.assertEqual(self.num_st, np.sum(n))

        n, b, p = self.clp.hist_r_storm(ax=ax, norm_x=True)
        self.assertEqual(self.num_st, np.sum(n))

    def test_hist_phi_storm(self):
        fig, ax = plt.subplots()
        n, b, p = self.clp.hist_phi_storm(ax=ax)
        self.assertEqual(self.num_poles, np.sum(n))
        plt.close()

    def test_misc(self):
        fig = self.clp.figure()
        self.clp.savefig('deleteme.png')
        plt.close()
