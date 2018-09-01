from colicoords.fileIO import load, save
from colicoords.preprocess import data_to_cells
from test.testcase import ArrayTestCase
from test.test_functions import load_testdata
import unittest
import os



class FileIOTest(ArrayTestCase):
    def setUp(self):
        self.data = load_testdata('ds3')
        self.cell_list = data_to_cells(self.data, initial_crop=2, rotate='binary')
        self.cell_obj = self.cell_list[0]
        self.cell_obj.optimize()

    def test_save_load_cells(self):
        save('temp_save.hdf5', self.cell_obj)
        cell_obj_load = load('temp_save.hdf5')
        os.remove('temp_save.hdf5')

        for item in ['r', 'xl', 'xr']:
            self.assertEqual(getattr(self.cell_obj.coords, item), getattr(cell_obj_load.coords, item))

        self.assertEqual(self.cell_obj.name, cell_obj_load.name)

        for p1, p2 in zip(self.cell_obj.coords.coeff, cell_obj_load.coords.coeff):
            self.assertEqual(p1, p2)

        self.assertArrayEqual(self.cell_obj.data.binary_img, cell_obj_load.data.binary_img)
        self.assertArrayEqual(self.cell_obj.data.flu_fluorescence, cell_obj_load.data.flu_fluorescence)

    def test_save_load_cell_list(self):
        save('temp_save_celllist.hdf5', self.cell_list)
        cell_list_load = load('temp_save_celllist.hdf5')
        os.remove('temp_save_celllist.hdf5')

        self.assertEqual(len(self.cell_list), len(cell_list_load))

        for ci, co in zip(self.cell_list, cell_list_load):
            for item in ['r', 'xl', 'xr']:
                self.assertEqual(getattr(ci.coords, item), getattr(co.coords, item))

            self.assertEqual(ci.name, co.name)

            for p1, p2 in zip(ci.coords.coeff, co.coords.coeff):
                self.assertEqual(p1, p2)

            self.assertArrayEqual(ci.data.binary_img, co.data.binary_img)
            self.assertArrayEqual(ci.data.flu_fluorescence, co.data.flu_fluorescence)


if __name__ == '__main__':
    unittest.main()