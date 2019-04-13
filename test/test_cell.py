from test.testcase import ArrayTestCase
from test.test_functions import load_testdata
from colicoords.preprocess import data_to_cells
from colicoords.data_models import Data
from colicoords.support import pad_cell
from colicoords.cell import CellList, Cell
from colicoords.fileIO import load
import os
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

        self.assertEqual(r_max, 9.0)
        self.assertAlmostEqual(r_mid, 6.49, 2)


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

    def test_l_classify(self):
        pass

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


if __name__ == '__main__':
    unittest.main()
