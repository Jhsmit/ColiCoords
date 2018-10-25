import unittest
import os
from symfit import Fit
from colicoords import load
from colicoords.models import PSF, RDistModel, Memory
from colicoords.fitting import LinearModelFit
from colicoords.minimizers import *


class RDistModelFittingTest(unittest.TestCase):
    def setUp(self):
        f_path = os.path.dirname(os.path.realpath(__file__))
        self.cells = load(os.path.join(f_path, 'test_data', 'test_cells.hdf5'))
        self.memory = Memory(verbose=0)
        self.de_kwargs = {'popsize': 10, 'seed': 42}

    def test_rdistmodel_fit(self):
        psf = PSF(sigma=1.59146972e+00)
        rm = RDistModel(psf, mem=self.memory, r='equal')
        x, y = self.cells[0].r_dist(20, 1)
        y -= y.min()

        fit = Fit(rm, x, y, minimizer=Powell, sigma_y=1/np.sqrt(y))
        res = fit.execute()

        par_dict = {'a1': 75984.78344557587, 'a2': 170938.0835695505, 'r': 7.186390052694122}
        for k, v in par_dict.items():
            self.assertAlmostEqual(v, res.params[k], 2)
        self.assertAlmostEqual(21834555979.09033, res.objective_value, 3)

        fit = Fit(rm, x, y, minimizer=Powell)
        res = fit.execute()

        par_dict = {'a1': 86129.37542153012, 'a2': 163073.91919617794, 'r': 7.372535479080642}
        for k, v in par_dict.items():
            self.assertAlmostEqual(v, res.params[k], 2)
        self.assertAlmostEqual(7129232.534842306, res.objective_value, 3)

    def test_linear_fit(self):
        psf = PSF(sigma=1.59146972e+00)
        rm = RDistModel(psf, mem=self.memory, r='equal')
        x, y = self.cells[0].r_dist(20, 1)
        y -= y.min()

        fit = LinearModelFit(rm, x, y, minimizer=Powell, sigma_y=1/np.sqrt(y))
        res = fit.execute()

        par_dict = {'a1': 75355.7394237, 'a2': 172377.87770918, 'r': 7.206538881423469}
        for k, v in par_dict.items():
            self.assertAlmostEqual(v, float(res.params[k]), 2)
        self.assertAlmostEqual(22329654459.541504, float(res.objective_value), 3)

        fit = LinearModelFit(rm, x, y, minimizer=Powell)
        res = fit.execute()

        par_dict = {'a1': 86129.31724353383, 'a2': 163073.9713967057, 'r': 7.372535479080642}
        for k, v in par_dict.items():
            self.assertAlmostEqual(v, float(res.params[k]), 2)
        self.assertAlmostEqual(7129232.534804644, float(res.objective_value), 5)

    def test_linear_fit_global(self):
        #todo repeat with generated data where global fitting acutally makes sense
        psf = PSF(sigma=1.59146972e+00)
        rm = RDistModel(psf, mem=self.memory, r='equal')
        x, y = self.cells[0:10].r_dist(20, 1)
        y_min = np.min(y, axis=1)
        y -= y_min[:, np.newaxis]

        fit = LinearModelFit(rm, x, y, minimizer=DifferentialEvolution)
        res = fit.execute(**self.de_kwargs)

        self.assertAlmostEqual(345643270.9300041, res.objective_value, 3)

        fit = LinearModelFit(rm, x, y, minimizer=DifferentialEvolution, sigma_y=1/np.sqrt(y))
        res = fit.execute(**self.de_kwargs)

        self.assertAlmostEqual(3040393780457.542, res.objective_value, 1)


if __name__ == '__main__':
    unittest.main()