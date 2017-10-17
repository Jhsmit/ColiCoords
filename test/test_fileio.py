from colicoords.fileIO import load, save
from colicoords.cell import Cell
from colicoords.preprocess import data_to_cells
from testcase import ArrayTestCase
from test_functions import generate_testdata
import unittest
import tifffile
import numpy as np


class FileIOTest(ArrayTestCase):
    def setUp(self):
        self.data = generate_testdata('ds3')
        cell_list = data_to_cells(self.data, pad_width=2, cell_frac=0.5, rotate='Binary')
        self.cell_obj = cell_list[0]
        self.cell_obj.optimize()

    def test_save_load(self):
        save('temp_save.cc', self.cell_obj)
        cell_obj_load = load('temp_save.cc')

        for item in ['r', 'xl', 'xr']:
            self.assertEqual(getattr(self.cell_obj.coords, item), getattr(cell_obj_load.coords, item))

        self.assertEqual(self.cell_obj.label, cell_obj_load.label)

        for p1, p2 in zip(self.cell_obj.coords.coeff, cell_obj_load.coords.coeff):
            self.assertEqual(p1, p2)

        #todo these tests alwasys pass!!! (or dont they?) --> they do
        self.assertTrue(np.all(self.cell_obj.data.binary_img == cell_obj_load.data.binary_img))
        self.assertTrue(np.all(self.cell_obj.data.flu_Fluorescence == cell_obj_load.data.flu_Fluorescence))

if __name__ == '__main__':
    unittest.main()