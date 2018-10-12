import unittest
from test.testcase import ArrayTestCase
from test.test_functions import load_testdata
from colicoords import CellFit, load
from colicoords.models import PSF, RDistModel
from colicoords.minimizers import *

class TestCellFitting(ArrayTestCase):
    def setUp(self):
        self.cells = load(r'test_data/test_cells.hdf5')

    def test_fitting_binary(self):
        resdict = {'a0':2.004259e+01, 'a1':-2.571423e-01, 'a2':4.944874e-03,
                   'r':9.096484e+00, 'xl': 1.552373e+01, 'xr':3.613869e+01}

        cell_0 = self.cells[0].copy()
        res = cell_0.optimize()
        self.assertEqual(res.objective_value, 30)
        for key, val in resdict.items():
            self.assertAlmostEqual(res.params[key], val, 5)

        cell_0 = self.cells[0].copy()
        res = cell_0.optimize(minimizer=Powell)
        self.assertEqual(res.objective_value, 30)
        for key, val in resdict.items():
            self.assertAlmostEqual(res.params[key], val, 5)

        cell_0 = self.cells[0].copy()
        res = cell_0.optimize(minimizer=DifferentialEvolution)

        self.assertLessEqual(res.objective_value, 25)


    def test_fitting_brightfield(self):
        cell_0 = self.cells[0].copy()
        res = cell_0.optimize('brightfield')
        print(res)
        print(res.objective_value)

        cell_0 = self.cells[0].copy()
        res = cell_0.optimize('brightfield')
        print(res)
        print(res.objective_value)

        # print('binary powell', res, res.objective_value)
        # res = cell_0.optimize('brightfield')
        # print('bf', res, res.objective_value)
        # res = cell_0.optimize('brightfield', minimizer=Powell)
        # print('bf', res, res.objective_value)