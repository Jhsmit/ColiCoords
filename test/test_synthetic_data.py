from colicoords.synthetic_data import SynthCell
from test import testcase


import unittest
import numpy as np


class SynthCellTest(testcase.ArrayTestCase):
    def setUp(self):
        length = np.random.normal(40, 5)
        radius = np.random.normal(12, 2)
        curvature = np.random.normal(0, 0.005)

        self.cell_1 = SynthCell(length, radius, curvature)

        length = np.random.normal(40, 5)
        radius = np.random.normal(12, 2)
        curvature = np.random.normal(0, 0.005)

        self.cell_2 = SynthCell(length, radius, curvature)

    def test_coordtransform(self):
        x, y = self.cell_1.coords.rev_transform(self.cell_1.coords.lc,
                                                self.cell_1.coords.rc,
                                                self.cell_1.coords.phi, l_norm=False)

        lc, rc, phi = self.cell_1.coords.transform(x, y)

        self.assertArrayAlmostEqual(rc, self.cell_1.coords.rc)
        self.assertArrayAlmostEqual(lc, self.cell_1.coords.lc)
        self.assertArrayAlmostEqual(phi, self.cell_1.coords.phi)


if __name__ == '__main__':
    unittest.main()