from test.testcase import ArrayTestCase
from test.test_functions import load_testdata
from colicoords.synthetic_data import SynthCell
from colicoords.preprocess import data_to_cells
from colicoords.models import PSF, RDistModel
from colicoords.cell import CellList, Cell
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




if __name__ == '__main__':
    unittest.main()
