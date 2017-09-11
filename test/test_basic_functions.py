from cellcoordinates.fileIO import load, save
from cellcoordinates.cell import Cell, CellList
from cellcoordinates.preprocess import process_cell
from testcase import ArrayTestCase
from test_functions import generate_testdata
from cellcoordinates.gui.controller import CellObjectController
import unittest
import tifffile
import numpy as np


class CellTest(ArrayTestCase):
    def setUp(self):
        self.data = generate_testdata()

    def test_celllist(self):
        ctrl = CellObjectController(self.data, '')
        cell_list = ctrl._create_cell_objects(self.data, 0.5, 2, 'Binary')

        cl = CellList(cell_list)
        self.assertEqual(len(cl), 9)
        c5 = cl[5]
        self.assertIsInstance(c5, Cell)

        del cl[5]
        self.assertEqual(len(cl), 8)
        self.assertTrue(cl[3] in cl)

        vol = cl.volume
        self.assertEqual(len(vol), 8)
        print(vol)

if __name__ == '__main__':
    unittest.main()