from colicoords.fileIO import load
from colicoords.cell import Cell, CellList
from colicoords.support import pad_cell, crop_cell, pad_data, crop_data, label_stack, multi_dilate, multi_erode
from test.testcase import ArrayTestCase
from test.test_functions import load_testdata
import numpy as np
import mahotas as mh
import os

class TestData(ArrayTestCase):
    def setUp(self):
        self.data = load_testdata('ds1')
        f_path = os.path.dirname(os.path.realpath(__file__))
        self.cells = load(os.path.join(f_path, 'test_data/test_synth_cell_storm.hdf5'))
        self.cell = self.cells[0]

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

    def test_allow_scalars(self):
        r, l, phi = self.cell.coords.transform(10, 20)
        self.assertTrue(np.isscalar(r))
        self.assertTrue(np.isscalar(l))
        self.assertTrue(np.isscalar(phi))


class TestImgProcess(ArrayTestCase):
    def setUp(self):
        self.data = load_testdata('ds1')

    def test_label(self):
        binary = self.data.binary_img.astype(bool)
        labelled = label_stack(binary)
        for l, b in zip(labelled, binary):
            l, n = mh.label(b)
            self.assertEqual(len(np.unique(l)), n + 1)

    def test_erode_dilate(self):
        binary = self.data.binary_img.astype(bool)[0]
        s = np.sum(binary)
        e = multi_erode(binary, 0)
        self.assertEqual(np.sum(e), s)

        eroded = multi_erode(binary, 1)
        self.assertLess(np.sum(eroded), s)
        self.assertArrayEqual(mh.erode(binary), eroded)

        dilated = multi_dilate(binary, 1)
        self.assertGreater(np.sum(dilated), s)
        self.assertArrayEqual(mh.dilate(binary), dilated)

        eroded = multi_erode(binary, 100)
        self.assertEqual(eroded.sum(), 0)

        dilated = multi_dilate(binary, 300)
        self.assertEqual(dilated.sum(), np.product(binary.shape))

    def test_temmp(self):
        binary = self.data.binary_img.astype(bool)[0]

        eroded = multi_erode(binary, 1)
        self.assertLess(np.sum(eroded), s)
        self.assertArrayEqual(mh.erode(binary), e)
