from colicoords.data_models import Data
from colicoords.fileIO import load
from colicoords.cell import Cell, CellList
from colicoords.support import pad_cell, crop_cell, pad_data, crop_data
from test.testcase import ArrayTestCase
from test.test_functions import load_testdata
import numpy as np
import os

class TestData(ArrayTestCase):
    def setUp(self):
        self.data = load_testdata('ds1')
        f_path = os.path.dirname(os.path.realpath(__file__))
        self.cells = load(os.path.join(f_path, 'test_data/test_synth_cell_storm.hdf5'))

    def test_pad(self):
        shape_0, shape_1 = zip(*[cell_obj.data.shape for cell_obj in self.cells])
        shape_0_max, shape_1_max = np.max(shape_0), np.max(shape_1)
        cell_list = CellList([pad_cell(cell_obj, (shape_0_max, shape_1_max)) for cell_obj in self.cells])
        for cell in cell_list:
            self.assertEqual((shape_0_max, shape_1_max), cell.data.shape)
            self.assertEqual((shape_0_max, shape_1_max), cell.coords.shape)

        # pad by zero and check if the results are the same
        x, y = self.cells.r_dist(20, 1, method='box')
        cell_list = CellList([pad_cell(cell_obj, cell_obj.data.shape) for cell_obj in self.cells])
        xn, yn = cell_list.r_dist(20, 1, method='box')
        self.assertArrayEqual(y, yn)

        #pad by known amount and check of storm coords more accordingly
        cell = self.cells[0]
        shape_0, shape_1 = cell.data.shape
        padded_cell = pad_cell(cell, (shape_0 + 10, shape_1 + 10))
        self.assertArrayEqual(cell.data.data_dict['storm']['x'] + 5, padded_cell.data.data_dict['storm']['x'])

    def test_crop(self):
        shape_0, shape_1 = 20, 30
        cell_list = CellList([crop_cell(cell_obj, (shape_0, shape_1)) for cell_obj in self.cells])
        for cell in cell_list:
            self.assertEqual((shape_0, shape_1), cell.data.shape)
            self.assertEqual((shape_0, shape_1), cell.coords.shape)