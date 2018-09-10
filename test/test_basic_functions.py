from colicoords.cell import Cell, CellList
from test import testcase
from test.test_functions import load_testdata
import hashlib
from colicoords.preprocess import data_to_cells
import unittest


class DataTest(testcase.ArrayTestCase):
    def setUp(self):
        self.data = load_testdata('ds1')

    def test_data_slicing(self):
        print(self.data.shape)
        sl1 = self.data[2:5, :, :]
        self.assertEqual(sl1.shape, (3, 512, 512))

        sl2 = self.data[:, 20:40, 100:200]
        self.assertEqual(sl2.shape, (10, 20, 100))

    def test_data_copy(self):
        m0 = self.data.binary_img.mean()
        data_copy = self.data.copy()
        self.assertEqual(m0, self.data.binary_img.mean())
        data_copy.binary_img += 20

        self.assertEqual(m0, self.data.binary_img.mean())
        self.assertEqual(data_copy.binary_img.mean(), m0 + 20)

    def _test_cell_list(self):
        #todo check order
        print(hashlib.md5(self.data).hexdigest())
        cell_list = data_to_cells(self.data, initial_crop=2, cell_frac=0.5, rotate='binary')
        print(hashlib.md5(self.data).hexdigest())
        cell_list = data_to_cells(self.data, initial_crop=2, cell_frac=0.5, rotate='binary')
        print(hashlib.md5(self.data).hexdigest())

        d = self.data.copy()
        print(d == self.data)

        cl = CellList(cell_list)
        self.assertEqual(len(cl), 48)
        c5 = cl[5]
        self.assertIsInstance(c5, Cell)

        del cl[5]
        self.assertEqual(len(cl), 47)
        self.assertTrue(cl[3] in cl)
        cl.append(c5)
        self.assertTrue(c5 in cl)

        vol = cl.volume
        self.assertEqual(len(vol), 48)


class CellListTest(testcase.ArrayTestCase):
    def setUp(self):
        data = load_testdata('ds1')
        self.cell_list = data_to_cells(data)

    def test_slicing(self):
        sliced = self.cell_list[:5]
        self.assertIsInstance(sliced, CellList)


if __name__ == '__main__':
    unittest.main()