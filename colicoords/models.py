import numpy as np
from colicoords.optimizers import Parameter
from scipy.integrate import quad
from colicoords.config import cfg


try:
    from joblib import Memory as JobMemory
    class Memory(JobMemory):
        def __init__(self, *args, **kwargs):
            args = (cfg.CACHE_DIR,) + args
            super(Memory, self).__init__(*args, **kwargs)

except (ImportError, ModuleNotFoundError):
    'Package joblib not found, cached memory not available'


class PSF(object):
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, x):
        return (1/(self.sigma*np.sqrt(2*np.pi))) * np.exp(-(x/self.sigma)**2 / 2)

#todo psf is an object and depending on which instance it is joblib will recalculate even if the sigma is the same
#todo https://docs.scipy.org/doc/scipy/reference/generated/scipy.LowLevelCallable.html#scipy.LowLevelCallable
def _y1(x, r1, psf, psf_uid):
    def integrant(x, v, r1, psf):
        return psf(x - v) * np.nan_to_num(np.sqrt(r1 ** 2 - x ** 2))

    yarr, yerr = np.array([quad(integrant, -np.inf, np.inf, args=(v, r1, psf)) for v in x]).T
    return yarr


def _y2(x, r2, psf, psf_uid):
    def integrant(x, v, r2, psf):
        try:
            return psf(x - v) * np.nan_to_num(np.sqrt(1 + (x ** 2 / (r2 ** 2 - x ** 2))))
        except ZeroDivisionError:
            return 0

    yarr, yerr = np.array([quad(integrant, -np.inf, np.inf, args=(v, r2, psf)) for v in x]).T
    return yarr


class RDistModel(object):
    parameters = 'a1 r1 r2 a2'
    linear_parameters = 'a1 a2'

    def __init__(self, psf, mem=None):
        self.psf = psf

        self.r1 = Parameter(name='r1', value=4.5, min=2, max=6)
        self.a1 = Parameter(name='a1', value=0.5, min=0)
        self.r2 = Parameter(name='r2', value=5.5, min=2, max=8)
        self.a2 = Parameter(name='a2', value=0.5, min=0)

        self.yerr = None

        if mem is not None:
            self.y1 = mem.cache(_y1, ignore=['psf'])
            self.y2 = mem.cache(_y2, ignore=['psf'])
        else:
            self.y1 = _y1
            self.y2 = _y2

        self.i = 10

    def __call__(self, x, **kwargs):
        r1 = kwargs.pop('r1', self.r1.value)
        r2 = kwargs.pop('r2', self.r2.value)
        a1 = kwargs.pop('a1', self.a1.value)
        a2 = kwargs.pop('a2', self.a2.value)

        if self.i:
            r1_l = int(np.floor(self.i * r1)) / self.i
            r1_u = int(np.ceil(self.i * r1)) / self.i

            if r1_l == r1 and r1_u == r1:
                y1 = self.y1(x, r1, self.psf, self.psf.sigma)
            else:
                y1_l = self.y1(x, r1_l, self.psf, self.psf.sigma)
                y1_u = self.y1(x, r1_u, self.psf, self.psf.sigma)

                y1 = (y1_l*(r1_u - r1) + y1_u*(r1 - r1_l)) / (r1_u - r1_l)

            r2_l = int(np.floor(self.i * r2)) / self.i
            r2_u = int(np.ceil(self.i * r2)) / self.i

            if r2_l == r2 and r2_u == r2:
                y2 = self.y2(x, r2, self.psf, self.psf.sigma)
            else:
                y2_l = self.y2(x, r2_l, self.psf, self.psf.sigma)
                y2_u = self.y2(x, r2_u, self.psf, self.psf.sigma)

                y2 = (y2_l * (r2_u - r2) + y2_u * (r2 - r2_l)) / (r2_u - r2_l)

        else:
            y1 = self.y1(x, r1, self.psf, self.psf.sigma)
            y2 = self.y2(x, r2, self.psf, self.psf.sigma)

        yarr = (a1 / (0.5 * np.pi * r1 ** 2))*y1 + (a2 / (np.pi * r2))*y2

        return yarr

    def get_bounds(self, parameters, bounded):
        bounds = [(getattr(self, par).min, getattr(self, par).max) if par in bounded.split(' ') else
                  (None, None) for par in parameters.split(' ')]

        if len(bounds) == 0:
            return None
        elif np.all(np.array(bounds) == (None, None)):
            return None
        else:
            return bounds

    def get_constraints(self, parameters):
        def _constr(par_values, par_names):
            par_dict = {par_name: par_value for par_name, par_value in zip(par_names, par_values)}
            return par_dict['r2'] - par_dict['r1']

        if 'r1' in parameters and 'r2' in parameters:
            constraints = {'type': 'ineq', 'fun': _constr, 'args': (parameters.split(' '), )}
        else:
            constraints = None

        return constraints

    def sub_par(self, res_dict):

        for k, v in res_dict.items():
            getattr(self, k).value = v
