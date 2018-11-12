import unittest
import os
from symfit import Fit, Eq
from test.testcase import ArrayTestCase
from colicoords import CellFit, load
from colicoords.minimizers import *
import platform

class TestCellFitting(ArrayTestCase):
    def setUp(self):
        f_path = os.path.dirname(os.path.realpath(__file__))
        self.cells = load(os.path.join(f_path, 'test_data', 'test_cells.hdf5'))
        self.de_kwargs = {'popsize': 10, 'recombination': 0.9, 'seed': 42}

    def test_fitting_binary(self):
        resdict = {'a0': 2.004259e+01, 'a1': -2.571423e-01, 'a2': 4.944874e-03,
                   'r': 9.096484e+00, 'xl': 1.552373e+01, 'xr': 3.613869e+01}

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
        res = cell_0.optimize(minimizer=DifferentialEvolution, **self.de_kwargs)

        self.assertEqual(res.objective_value, 24)

    def test_fitting_brightfield(self):
        res_dict = {'a0': 2.059920e+01, 'a1': -2.712269e-01, 'a2': 5.158162e-03,
                    'r': 9.248204e+00, 'xl': 1.522411e+01, 'xr': 3.645077e+01}

        cell_0 = self.cells[0].copy()
        res = cell_0.optimize('brightfield')

        bf_value = 10016887.213015744 if platform.system() == 'Linux' else 10016887.123816863

        self.assertEqual(res.objective_value, bf_value)
        for key, val in res_dict.items():
            self.assertAlmostEqual(res.params[key], val, 5)

        cell_0 = self.cells[0].copy()
        res = cell_0.optimize('brightfield', minimizer=Powell)
        self.assertEqual(res.objective_value, bf_value)
        for key, val in res_dict.items():
            self.assertAlmostEqual(res.params[key], val, 5)

    def test_fitting_brightfield_DE(self):
        cell_0 = self.cells[0].copy()
        res_dict = {'a0': 1.762122e+01, 'a1': -2.272049e-02, 'a2': 4.457729e-04,
                    'r': 9.243795e+00, 'xl': 1.517070e+01, 'xr': 3.638401e+01}

        res = cell_0.optimize('brightfield', minimizer=DifferentialEvolution, **self.de_kwargs)
        self.assertAlmostEqual(res.objective_value, 8206601.073967202, 6)
        for k, v in res_dict.items():
            self.assertAlmostEqual(v, res.params[k], 5)

    def test_cell_list_optimize(self):
        cells = self.cells[:8].copy()
        res_list = cells.optimize()

        # Check if the result has been properly substituted in all cell objects
        for r, cell in zip(res_list, cells):
            for k, v in r.params.items():
                self.assertEqual(v, getattr(cell.coords, k))

        obj_values = [30, 25, 26, 25, 26, 22, 37, 17]
        for r, val in zip(res_list, obj_values):
            self.assertEqual(r.objective_value, val)

        cells = self.cells[:8].copy()
        res_list = cells.optimize(minimizer=Powell)

        # Check if the result has been properly substituted in all cell objects
        for r, cell in zip(res_list, cells):
            for k, v in r.params.items():
                self.assertEqual(v, getattr(cell.coords, k))

        obj_values = [30, 25, 26, 25, 26, 22, 37, 17]
        for r, val in zip(res_list, obj_values):
            self.assertEqual(r.objective_value, val)

    def test_cell_list_optimize_pertubation(self):
        cells = self.cells[:8].copy()
        for cell in cells:
            cell.coords.xl -= 3
            cell.coords.xr += 3
            cell.coords.a0 *= 0.95
            cell.coords.a1 *= 1.05

        res_list = cells.optimize(minimizer=Powell)

        # Check if the result has been properly substituted in all cell objects
        for r, cell in zip(res_list, cells):
            for k, v in r.params.items():
                self.assertEqual(v, getattr(cell.coords, k))

        obj_values = [28, 26, 24, 53, 31, 20, 46, 11]
        for r, val in zip(res_list, obj_values):
            self.assertEqual(r.objective_value, val)

    def test_multiprocessing(self):
        cells = self.cells[:8].copy()
        res_list = cells.optimize_mp()

        # Check if the result has been properly substituted in all cell objects
        for r, cell in zip(res_list, cells):
            for k, v in r.params.items():
                self.assertEqual(v, getattr(cell.coords, k))

        obj_values = [30, 25, 26, 25, 26, 22, 37, 17]
        for r, val in zip(res_list, obj_values):
            self.assertEqual(r.objective_value, val)

        cells = self.cells[:8].copy()
        res_list = cells.optimize_mp(minimizer=Powell)

        # Check if the result has been properly substituted in all cell objects
        for r, cell in zip(res_list, cells):
            for k, v in r.params.items():
                self.assertEqual(v, getattr(cell.coords, k))

        obj_values = [30, 25, 26, 25, 26, 22, 37, 17]
        for r, val in zip(res_list, obj_values):
            self.assertEqual(r.objective_value, val)

    def test_multiprocessing_brightfield(self):
        cells = self.cells[:8].copy()
        res_list = cells.optimize_mp('brightfield')

        # Check if the result has been properly substituted in all cell objects
        for r, cell in zip(res_list, cells):
            for k, v in r.params.items():
                self.assertEqual(v, getattr(cell.coords, k))

        if platform.system() == 'Linux':
            obj_values = [10016887.213015744, 23617786.724680573, 8999333.060823418, 29395182.431100346,
                          62892422.38607424, 20011819.274376377, 33025293.22872089, 112600585.35048027]

        else:
            obj_values = [10016887.123816863, 23617786.697511297, 8999333.084250152, 29339637.49970112,
                          62892422.65259473, 20011819.287397716, 33025293.172053672, 112600585.34296474]

        for r, val in zip(res_list, obj_values):
            self.assertAlmostEqual(r.objective_value, val, 5)

    # def test_temp(self):
    #     cells = self.cells[:4].copy()
    #     #perturb inital coordinate guesses
    #     for cell in cells:
    #         cell.coords.xl -= 3
    #         cell.coords.xr += 3
    #         cell.coords.a0 *= 0.9
    #         cell.coords.a1 *= 1.05
    #
    #     res_list = cells.optimize_mp(minimizer=DifferentialEvolution, **self.de_kwargs)
    #
    #     # Check if the result has been properly substituted in all cell objects
    #     for r, cell in zip(res_list, cells):
    #         for k, v in r.params.items():
    #             self.assertEqual(v, getattr(cell.coords, k))
    #
    #     obj_values = [24.0, 17.0, 19.0, 22.0]
    #     for r, val in zip(res_list, obj_values):
    #         self.assertEqual(r.objective_value, val)


class TestSynthCellFitting(ArrayTestCase):
    def setUp(self):
        f_path = os.path.dirname(os.path.realpath(__file__))
        self.cells = load(os.path.join(f_path, 'test_data', 'test_synth_cell_storm.hdf5'))

    def test_fitting_storm(self):
        cell = self.cells[0]
        cell_fit = cell.copy()

        cell_fit.coords.xl -= 5
        cell_fit.coords.xr -= 2
        cell_fit.coords.a0 -= 5
        cell_fit.coords.a1 *= 0.1
        cell_fit.coords.a2 *= -1.2
        cell_fit.coords.r += 5

        res = cell_fit.optimize('storm', minimizer=Powell)

        res_dict = {'a0': 10.97109892275029, 'a1': 0.18567779050907443, 'a2': -0.0025215258088723746,
                    'r': 7.4819827565860555, 'xl': 13.174377724643914, 'xr': 60.56108371075396}

        self.assertAlmostEqual(214.05394964376524, res.objective_value, 5)
        for k, v in res_dict.items():
            self.assertAlmostEqual(v, res.params[k], 3)


if __name__ == '__main__':
    unittest.main()
