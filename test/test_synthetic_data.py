from colicoords.synthetic_data import SynthCell
from colicoords.models import PSF, RDistModel
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


class TestSynthCell(testcase.ArrayTestCase):
    def setUp(self):
        self.cell_obj = SynthCell(40, 12, 0.01)  # length, radius, curvature
        psf = PSF(1.54)
        rmodel = RDistModel(psf)
        parameters = {'a1': 0.0, 'a2': 0.1, 'r1': 6, 'r2': 6.5}

        img = self.cell_obj.gen_radial_model_data(rmodel, parameters)
        self.cell_obj.data.add_data(img, 'fluorescence')

    def test_measure_r(self):
        r_max = self.cell_obj.measure_r(data_name='fluorescence', mode='max', in_place=False, step=0.5)
        r_mid = self.cell_obj.measure_r(data_name='fluorescence', mode='mid', in_place=False)

        self.assertEqual(r_max, 5.0)
        self.assertAlmostEqual(r_mid, 7.58, 2)


if __name__ == '__main__':
    unittest.main()