import unittest
from test.testcase import ArrayTestCase
from test.test_functions import load_testdata
from colicoords import SynthCell, data_to_cells
from colicoords.models import PSF, RDistModel


class TestCell(ArrayTestCase):
    def setUp(self):
        #todo update this
        self.data = load_testdata('ds3')
        self.cell_list = data_to_cells(self.data, initial_crop=2, rotate='binary')
        self.cell_obj = self.cell_list[0]
        self.cell_obj.optimize()

    def test_measure_r(self):
        r_max = self.cell_obj.measure_r(data_name='fluorescence', mode='max', in_place=False, step=0.5)
        r_mid = self.cell_obj.measure_r(data_name='fluorescence', mode='mid', in_place=False)

        self.assertEqual(r_max, 5.0)
        self.assertAlmostEqual(r_mid, 8.11, 2)

        r_max = self.cell_obj.measure_r(data_name='brightfield', mode='max', in_place=False, step=0.5)
        r_mid = self.cell_obj.measure_r(data_name='brightfield', mode='mid', in_place=False)

        self.assertEqual(r_max, 9.0)
        self.assertAlmostEqual(r_mid, 6.49, 2)


class TestSynthCell(ArrayTestCase):
    def setUp(self):
        self.cell_obj = SynthCell(40, 12, 0.01)  # length, radius, curvature
        psf = PSF(1.54)
        rmodel = RDistModel(psf)
        parameters = {'a1': 0.0, 'a2': 0.1, 'r1': 6, 'r2': 6.5}

        self.cell_obj.add_radial_model_data(rmodel, parameters)

    def test_measure_r(self):
        r_max = self.cell_obj.measure_r(data_name='fluorescence', mode='max', in_place=False, step=0.5)
        r_mid = self.cell_obj.measure_r(data_name='fluorescence', mode='mid', in_place=False)

        self.assertEqual(r_max, 5.0)
        self.assertAlmostEqual(r_mid, 7.58, 2)


if __name__ == '__main__':
    unittest.main()
