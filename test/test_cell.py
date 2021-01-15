from test.testcase import ArrayTestCase
from test.test_functions import load_testdata
from colicoords.preprocess import data_to_cells
from colicoords.data_models import Data
from colicoords.support import pad_cell
from colicoords.cell import CellList, Cell
from colicoords.fileIO import load
import os
import sys
import numpy as np
import unittest


class TestCell(ArrayTestCase):
    def setUp(self):
        #todo update this
        self.data = load_testdata('ds2')
        self.cell_list = data_to_cells(self.data, initial_crop=2, rotate='binary')
        self.cell_obj = self.cell_list[0]
        self.cell_obj.optimize()

    def test_measure_r(self):
        r_max = self.cell_obj.measure_r(data_name='fluorescence', mode='max', in_place=False, step=0.5)
        r_mid = self.cell_obj.measure_r(data_name='fluorescence', mode='mid', in_place=False)

        self.assertEqual(r_max, 5.0)
        self.assertAlmostEqual(r_mid, 8.11, 2)

        r_max = self.cell_obj.measure_r(data_name='brightfield', mode='max', in_place=False, step=0.5)
        r_mid = self.cell_obj.measure_r(data_name='brightfield', mode='mid', in_place=False)
        r_min = self.cell_obj.measure_r(data_name='brightfield', mode='min', in_place=False)
        print(r_min)

        self.assertEqual(r_max, 9.0)
        self.assertAlmostEqual(r_mid, 6.49, 2)

        with self.assertRaises(ValueError):
            r_ = self.cell_obj.measure_r(mode='asdf')

        cell = self.cell_obj.copy()
        r_max = cell.measure_r(data_name='brightfield', mode='max', in_place=True, step=0.5)
        self.assertEqual(r_max, None)
        self.assertEqual(cell.coords.r, 9.0)

    def test_reconstruct(self):
        bf_recontstr = self.cell_obj.reconstruct_image('brightfield')
        lsq = np.sum((bf_recontstr - self.cell_obj.data.bf_img)**2)

        if sys.version_info.minor == 6:
            self.assertAlmostEqual(44728880.48196769, float(lsq), 2)
        else:
            # Changed from 44728880.4819674 between py3.6 -> py3.6+
            self.assertAlmostEqual(44774714.40809806, float(lsq), 2)

        bf_rscl = self.cell_obj.reconstruct_image('brightfield', r_scale=0.5)
        cell = self.cell_obj.copy()
        cell.data.add_data(bf_rscl, 'brightfield', 'rescaled')
        r_mid = cell.measure_r(data_name='rescaled', mode='mid', in_place=False)
        self.assertAlmostEqual(12.974043291957795, float(r_mid), 2)

    def test_get_intensity(self):
        cell = self.cell_obj.copy()
        i0 = self.cell_obj.get_intensity()
        i1 = self.cell_obj.get_intensity(data_name='fluorescence')
        i2 = self.cell_obj.get_intensity(data_name='fluorescence', mask='coords')
        cell.coords.r *= 2
        i3 = cell.get_intensity(data_name='fluorescence', mask='coords')
        i4 = self.cell_obj.get_intensity(data_name='fluorescence', func=np.max)
        i5 = self.cell_obj.get_intensity(data_name='fluorescence', func=np.min)
        i6 = self.cell_obj.get_intensity(data_name='fluorescence', func=np.median)

        with self.assertRaises(ValueError):
            self.cell_obj.get_intensity(data_name='asdfsdfa')

        ii = np.array([i0, i1, i2, i3, i4, i5, i6])
        vi = np.array([23729.91051454139, 23729.91051454139, 23580.72807991121, 11281.533678756477, 40733, 3094, 27264.0])
        assert np.allclose(ii, vi)


class TestCellList(ArrayTestCase):
    def setUp(self):
        #todo update this
        self.data = load_testdata('ds2')
        self.cell_list = data_to_cells(self.data, initial_crop=2, rotate='binary')
        self.cell_obj = self.cell_list[0]
        self.cell_obj.optimize()

    def test_indexing(self):
        cell_list = self.cell_list[2:10]
        self.assertIsInstance(cell_list, CellList)

        cell = self.cell_list[5]
        self.assertEqual(np.where(self.cell_list.name == cell.name)[0][0], 5)
        self.assertIsInstance(cell, Cell)

        self.assertTrue(cell in self.cell_list)

    def test_geometry(self):
        props = ['radius', 'length', 'circumference', 'area', 'surface', 'volume']
        cell_list_copy = self.cell_list.copy()
        for prop in props:
            m1 = getattr(self.cell_list, prop)
            m2 = getattr(cell_list_copy, prop)

            self.assertArrayEqual(m1, m2)

        shape = self.cell_obj.data.shape
        cell_pad = pad_cell(self.cell_obj, (shape[0] + 5, shape[1] + 10))
        for prop in props:
            m1 = getattr(self.cell_obj, prop)
            m2 = getattr(cell_pad, prop)

            self.assertAlmostEqual(m1, m2, 6)  # On Linux (Travis) the result is exactly equal

# TODO 3D array fluorescence testing
class TestCellListSTORM(ArrayTestCase):
    def setUp(self):
        f_path = os.path.dirname(os.path.realpath(__file__))
        self.cell_list = load(os.path.join(f_path, 'test_data', 'test_synth_cell_storm.hdf5'))
        self.cell = self.cell_list[0]

        x = np.arange(20)
        y = np.exp(-x / 5)

        img_3d = self.cell.data.data_dict['fluorescence'][np.newaxis, :, :] * y[:, np.newaxis, np.newaxis]
        self.cell.data.add_data(img_3d, 'fluorescence', 'flu_3d')


        data = Data()
        data.add_data(self.cell.data.binary_img, 'binary')
        self.empty_cell = Cell(data)

    def test_l_dist(self):
        nbins = 50
        with self.assertRaises(IndexError):
            x, y = self.empty_cell.l_dist(nbins)
        with self.assertRaises(ValueError):  # todo refactor to stay as KeyError?
            x, y = self.cell.l_dist(nbins, data_name='notexisting')
        with self.assertRaises(ValueError):
            x, y = self.cell.l_dist(nbins, method='notexisting')

        storm_int_sum = np.sum(self.cell.data.data_dict['storm']['intensity'])

        x, y = self.cell.l_dist(nbins, data_name='storm', r_max=20)
        self.assertEqual(np.sum(y), storm_int_sum)

        x, y = self.cell.l_dist(nbins, data_name='storm', method='box', r_max=20)
        self.assertEqual(np.sum(y), storm_int_sum)

        x, y = self.cell.l_dist(nbins, data_name='storm', method='box', storm_weight=True, r_max=20)
        self.assertEqual(np.sum(y), storm_int_sum)

        x, y = self.cell.l_dist(nbins, data_name='fluorescence')

        x, y = self.cell.l_dist(nbins, data_name='fluorescence', method='box')

        x, y = self.cell.l_dist(nbins, method='box',)
        x, y = self.cell.l_dist(nbins, method='box', l_mean=0.75*self.cell.length, norm_x=True)
        x, y = self.cell.l_dist(nbins, method='box', norm_x=True)
        x, y = self.cell.l_dist(nbins, method='box', r_max=np.inf)
        x, y = self.cell.l_dist(nbins, method='box', r_max=1)
        x, y = self.cell.l_dist(nbins, method='box', r_max=0)

        x, y_list = self.cell.l_dist(nbins, data_name='flu_3d')
        self.assertEqual(len(y_list), 20)


    def test_l_classify(self):
        p, b, m = self.cell.l_classify(data_name='storm')
        total = len(self.cell.data.data_dict['storm'])
        self.assertEqual(p + b + m, total)

        p, b, m = self.cell.l_classify()
        total = len(self.cell.data.data_dict['storm'])
        self.assertEqual(p + b + m, total)

    def test_r_dist(self):
            stop = 15
            step = 0.5
            with self.assertRaises(IndexError):
                x, y = self.empty_cell.r_dist(stop, step)
            with self.assertRaises(ValueError):  # todo refactor to stay as KeyError?
                x, y = self.cell.r_dist(stop, step, data_name='notexisting')
            with self.assertRaises(ValueError):
                x, y = self.cell.r_dist(stop, step, method='notexisting')

            stop = 15
            step = 0.5
            bins_box = np.arange(0, stop + step, step) + 0.5 * step
            bins = np.arange(0, stop + step, step)
            storm_int_sum = np.sum(self.cell.data.data_dict['storm']['intensity'])

            x, y = self.cell.r_dist(data_name='storm', stop=stop, step=step)
            self.assertArrayEqual(bins_box, x)
            self.assertEqual(np.sum(y), storm_int_sum)

            x, y = self.cell.r_dist(data_name='storm', stop=stop, step=step, storm_weight=True)
            self.assertArrayEqual(bins_box, x)
            self.assertEqual(np.sum(y), storm_int_sum)

            x, y = self.cell.r_dist(stop, step, data_name='fluorescence')
            self.assertArrayEqual(bins, x)

            x, y = self.cell.r_dist(stop, step, data_name='fluorescence', limit_l='full')
            x, y = self.cell.r_dist(stop, step, data_name='fluorescence', limit_l='poles')

            with self.assertRaises(AssertionError):
                x, y = self.cell.r_dist(stop, step, data_name='fluorescence', limit_l=0)
            with self.assertRaises(AssertionError):
                x, y = self.cell.r_dist(stop, step, data_name='fluorescence', limit_l=1)

            x, y = self.cell.r_dist(stop, step, data_name='fluorescence', limit_l=0.5)
            self.assertArrayEqual(bins, x)
            x, y = self.cell.r_dist(stop, step, data_name='fluorescence', limit_l=1e-3)
            self.assertArrayEqual(bins, x)
            x, y = self.cell.r_dist(stop, step, data_name='fluorescence', limit_l=1e-10)
            self.assertArrayEqual(bins, x)
            x, y = self.cell.r_dist(stop, step, data_name='fluorescence', limit_l=1-1e-10)
            self.assertArrayEqual(bins, x)

    def test_phi_dist(self):
        step = 0.5
        with self.assertRaises(IndexError):
            x, yl, yr = self.empty_cell.phi_dist(step)
        with self.assertRaises(ValueError):  # todo refactor to stay as KeyError?
            x, yl, yr = self.cell.phi_dist(step, data_name='notexisting')
        with self.assertRaises(ValueError):
            x, yl, yr = self.cell.phi_dist(step, method='notexisting')

        stop = 180
        bins_box = np.arange(0, stop + step, step) + 0.5 * step
        bins = np.arange(0, stop + step, step)

        x, y = self.cell.data.data_dict['storm']['x'], self.cell.data.data_dict['storm']['y']
        lc, rc, psi = self.cell.coords.transform(x, y)

        b = np.logical_and(psi != 0, psi != 180)  # only localizations at poles
        storm_int_sum = np.sum(self.cell.data.data_dict['storm']['intensity'][b])

        x, yl, yr = self.cell.phi_dist(step, data_name='storm', r_max=np.inf)
        self.assertArrayEqual(bins_box, x)
        self.assertEqual(np.sum(yl + yr), storm_int_sum)

        x, yl, yr = self.cell.phi_dist(step, data_name='storm', storm_weight=True, r_max=np.inf)
        self.assertArrayEqual(bins_box, x)
        self.assertEqual(np.sum(yl + yr), storm_int_sum)

        x, yl, yr = self.cell.phi_dist(step, data_name='storm', r_max=0)
        self.assertEqual(np.sum(yl + yr), 0)

        x, yl, yr = self.cell.phi_dist(step, data_name='fluorescence')
        self.assertArrayEqual(bins, x)

        x, yl, yr = self.cell.phi_dist(step, data_name='fluorescence', r_min=-5)
        x, yl, yr = self.cell.phi_dist(step, data_name='fluorescence', r_max=0)
        x, yl, yr = self.cell.phi_dist(step, data_name='fluorescence', r_max=0, r_min=5)
        self.assertEqual(np.sum(yl + yr), 0)


if __name__ == '__main__':
    unittest.main()
