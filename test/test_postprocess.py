from test.testcase import ArrayTestCase
import matplotlib.pyplot as plt
from colicoords.plot import CellPlot, CellListPlot
from colicoords.fileIO import load
from colicoords.synthetic_data import SynthCell
from colicoords.postprocess import align_cells
import os
import numpy as np


#todo include storm intensity field
class TestPostProcess(ArrayTestCase):
    def setUp(self):
        f_path = os.path.dirname(os.path.realpath(__file__))
        self.cell_list = load(os.path.join(f_path, 'test_data', 'test_synth_cell_storm.hdf5'))
        self.num = len(self.cell_list)
        self.num_st = np.sum([len(cell.data.data_dict['storm']) for cell in self.cell_list])
        self.clp = CellListPlot(self.cell_list)

        self.num_poles = 0
        self.num_05 = 0
        for c in self.cell_list:
            st_x, st_y = c.data.data_dict['storm']['x'], c.data.data_dict['storm']['y']
            l, r, phi = c.coords.transform(st_x, st_y)
            self.num_poles += ((l == 0).sum() + (l == c.length).sum())
            self.num_05 += np.sum((l > 0.25 * c.length) * (l < 0.75 * c.length))
        self.num_full = self.num_st - self.num_poles

        self.model_cell = SynthCell(50, 8, 1e-10)

    def test_align_cells(self):
        aligned = align_cells(self.model_cell, self.cell_list)
        x, y = aligned.data.data_dict['storm']['x'], aligned.data.data_dict['storm']['y']
        l, r, phi = aligned.coords.transform(x, y)

        self.assertEqual(self.num_st, len(x))
        num_poles = (l == 0).sum() + (l == aligned.length).sum()
        self.assertEqual(num_poles, self.num_poles)

