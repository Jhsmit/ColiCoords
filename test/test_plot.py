from test.testcase import ArrayTestCase
import matplotlib.pyplot as plt
from colicoords.data_models import Data
from colicoords.plot import CellPlot
from colicoords.fileIO import load
import os
import numpy as np
import unittest

class TestCellPlot(ArrayTestCase):
    def setUp(self):
        f_path = os.path.dirname(os.path.realpath(__file__))
        self.cell_list = load(os.path.join(f_path, 'test_data', 'test_synth_cell_storm.hdf5'))
        self.cell = self.cell_list[0]
        self.cp = CellPlot(self.cell)

        x = np.arange(20)
        y = np.exp(-x / 5)

        img_3d = self.cell.data.data_dict['fluorescence'][np.newaxis, :, :] * y[:, np.newaxis, np.newaxis]
        self.cell.data.add_data(img_3d, 'fluorescence', 'flu_3d')

    def test_plot_midline(self):
        fig, ax = plt.subplots()

        line = self.cp.plot_midline(ax=ax)

        x = np.linspace(self.cell.coords.xl, self.cell.coords.xr, 100)
        y = np.polyval(x, self.cell.coords.coeff)
        xl, yl = line.get_data()
        self.assertArrayEqual(y, yl)