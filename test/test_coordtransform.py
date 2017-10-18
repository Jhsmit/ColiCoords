from colicoords import data_to_cells, Cell
from test_functions import generate_testdata
from testcase import ArrayTestCase
import numpy as np
import unittest
import tifffile


class Test(ArrayTestCase):
    def setUp(self):
        data = generate_testdata('ds1')
        cell_list = data_to_cells(data)
        self.cell = cell_list[0]

    def test_coordtransform(self):
        x, y = np.arange(10)**2, np.arange(10)+20
        xt, yt = self.cell.coords.transform(x, y, src='cart', tgt='mpl')
        xt2, yt2 = self.cell.coords.transform(xt, yt, src='mpl', tgt='cart')

        self.assertArrayEqual(x, xt2)
        self.assertArrayEqual(y, yt2)

if __name__ ==  '__main__':
    unittest.main()