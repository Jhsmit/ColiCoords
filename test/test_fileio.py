from cellcoordinates.fileIO import load, save
from cellcoordinates.cell import Cell
from cellcoordinates.preprocess import process_cell
from testcase import ArrayTestCase
import unittest
import tifffile
import numpy as np


class FileIOTest(ArrayTestCase):
    def setUp(self):
        bin_img = tifffile.imread('test_data/binary1.tif')
        flu_img = tifffile.imread('test_data/flu1.tif')
        self.cell_obj = process_cell(rotate='binary', binary_img=bin_img, fl_data={'514': flu_img})

        self.cell_obj.optimize(method='binary')

    def test_save_load(self):
        save('temp_save.cc', self.cell_obj)
        cell_obj_load = load('temp_save.cc')

        #todo testcase where degree == 2
        for item in ['r', 'xl', 'xr']:
            self.assertEqual(getattr(self.cell_obj.coords, item), getattr(cell_obj_load.coords, item))

        self.assertEqual(self.cell_obj.label, cell_obj_load.label)

        for p1, p2 in zip(self.cell_obj.coords.coeff, cell_obj_load.coords.coeff):
            self.assertEqual(p1, p2)

        #todo these tests alwasys pass!!! (or dont they?) --> they do
        self.assertTrue(np.all(self.cell_obj.data.binary_img == cell_obj_load.data.binary_img))
        self.assertTrue(np.all(self.cell_obj.data.fl_img_514 == cell_obj_load.data.fl_img_514))

if __name__ == '__main__':
    unittest.main()