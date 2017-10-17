from colicoords.cell import Cell
import numpy as np
import unittest
import tifffile


class Test(unittest.TestCase):
    def setUp(self):
        self.cell = Cell(binary_img=tifffile.imread(r'test_data/binary1.tif'))

    def test_coordtransform(self):
        x, y = np.arange(10)**2, np.arange(10)+20
        xt, yt =


if __name__ ==  '__main__':
    unittest.main()