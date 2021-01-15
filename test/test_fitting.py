import unittest
import os
from test.testcase import ArrayTestCase
from colicoords.fileIO import load
from colicoords.minimizers import *
import platform
from distutils.version import StrictVersion
import scipy
import symfit


class TestCellFitting(ArrayTestCase):
    def setUp(self):
        f_path = os.path.dirname(os.path.realpath(__file__))
        self.cells = load(os.path.join(f_path, 'test_data', 'test_cells.hdf5'))
        self.de_kwargs = {'popsize': 10, 'recombination': 0.9, 'seed': 42}

    @property
    def cf(self):
        if StrictVersion(symfit.__version__) < StrictVersion('0.5.0'):
            return 1
        else:
            return 2

    def test_fitting_binary(self):
        resdict = {'a0': 2.004259e+01, 'a1': -2.571423e-01, 'a2': 4.944874e-03,
                   'r': 9.096484e+00, 'xl': 1.552373e+01, 'xr': 3.613869e+01}

        cell_0 = self.cells[0].copy()
        res = cell_0.optimize()
        self.assertEqual(self.cf*res.objective_value, 30)
        for key, val in resdict.items():
            self.assertAlmostEqual(res.params[key], val, 5)

        cell_0 = self.cells[0].copy()
        res = cell_0.optimize(minimizer=Powell)
        self.assertEqual(self.cf*res.objective_value, 30)
        for key, val in resdict.items():
            self.assertAlmostEqual(res.params[key], val, 5)

        cell_0 = self.cells[0].copy()
        res = cell_0.optimize(minimizer=DifferentialEvolution, **self.de_kwargs)

        self.assertEqual(self.cf*res.objective_value, 24)

    def test_fitting_brightfield(self):
        res_dict = {'a0': 2.059920e+01, 'a1': -2.712269e-01, 'a2': 5.158162e-03,
                    'r': 9.248204e+00, 'xl': 1.522411e+01, 'xr': 3.645077e+01}

        cell_0 = self.cells[0].copy()
        res = cell_0.optimize('brightfield')

        bf_value = 10016887

        self.assertAlmostEqual(self.cf*res.objective_value, bf_value, 0)
        for key, val in res_dict.items():
            self.assertAlmostEqual(res.params[key], val, 5)

        cell_0 = self.cells[0].copy()
        res = cell_0.optimize('brightfield', minimizer=Powell)
        self.assertAlmostEqual(self.cf*res.objective_value, bf_value, 0)
        for key, val in res_dict.items():
            self.assertAlmostEqual(res.params[key], val, 5)

    def test_fitting_brightfield_DE(self):
        cell_0 = self.cells[0].copy()
        if StrictVersion(scipy.__version__) < StrictVersion('1.2.0'):
            res_dict = {'a0': 1.762122e+01, 'a1': -2.272049e-02, 'a2': 4.457729e-04,
                        'r': 9.243795e+00, 'xl': 1.517070e+01, 'xr': 3.638401e+01}
            value = 8206601.073967202
        elif StrictVersion(scipy.__version__) < StrictVersion('1.4.0'):
            res_dict = {'a0': 1.804735702887220E+01, 'a1': -5.285406247108937E-02, 'a2': 9.726930458977879E-04,
                        'r': 9.265014573271854E+00, 'xl': 1.520349136432933E+01, 'xr': 3.637929146737385E+01}
            value = 8221057.758640378
        else:
            res_dict = {'a0': 1.772944105424428E+01, 'a1': -2.800104528377645E-02, 'a2': 4.916945792890149E-04,
                        'r': 9.254283346347917E+00, 'xl': 1.5172830307331035E+01, 'xr': 3.645007538899884E+01}
            value = 8191888.515532021

        res = cell_0.optimize('brightfield', minimizer=DifferentialEvolution, **self.de_kwargs)
        self.assertAlmostEqual(self.cf*res.objective_value, value, 6)

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
            self.assertEqual(self.cf*r.objective_value, val)

        cells = self.cells[:8].copy()
        res_list = cells.optimize(minimizer=Powell)

        # Check if the result has been properly substituted in all cell objects
        for r, cell in zip(res_list, cells):
            for k, v in r.params.items():
                self.assertEqual(v, getattr(cell.coords, k))

        obj_values = [30, 25, 26, 25, 26, 22, 37, 17]
        for r, val in zip(res_list, obj_values):
            self.assertEqual(self.cf*r.objective_value, val)

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
            self.assertEqual(self.cf*r.objective_value, val)

    def test_multiprocessing(self):
        cells = self.cells[:8].copy()
        res_list = cells.optimize_mp()

        # Check if the result has been properly substituted in all cell objects
        for r, cell in zip(res_list, cells):
            for k, v in r.params.items():
                self.assertEqual(v, getattr(cell.coords, k))

        obj_values = [30, 25, 26, 25, 26, 22, 37, 17]
        for r, val in zip(res_list, obj_values):
            self.assertEqual(self.cf*r.objective_value, val)

        cells = self.cells[:8].copy()
        res_list = cells.optimize_mp(minimizer=Powell)

        # Check if the result has been properly substituted in all cell objects
        for r, cell in zip(res_list, cells):
            for k, v in r.params.items():
                self.assertEqual(v, getattr(cell.coords, k))

        obj_values = [30, 25, 26, 25, 26, 22, 37, 17]
        for r, val in zip(res_list, obj_values):
            self.assertEqual(self.cf*r.objective_value, val)

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
            self.assertAlmostEqual(self.cf*r.objective_value, val, 0)


class TestSynthCellFitting(ArrayTestCase):
    def setUp(self):
        f_path = os.path.dirname(os.path.realpath(__file__))
        self.cells = load(os.path.join(f_path, 'test_data', 'test_synth_cell_storm.hdf5'))

    @property
    def cf(self):
        if StrictVersion(symfit.__version__) < StrictVersion('0.5.0'):
            return 1
        else:
            return 2

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

        self.assertAlmostEqual(214.05394964376524, self.cf*res.objective_value, 5)
        for k, v in res_dict.items():
            self.assertAlmostEqual(v, res.params[k], 3)


if __name__ == '__main__':
    unittest.main()
