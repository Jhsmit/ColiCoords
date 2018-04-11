import numpy as np
from colicoords.optimizers import Parameter


#todo generator for allowing to load in experimental data to
class PSF(object):
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, x):
        return (1/self.sigma*np.sqrt(2*np.pi)) * np.exp(-(x/self.sigma)**2 / 2)


class RDistModel(object):

    def __init__(self, psf, x_range, num_points=351):  #todo more options for psf modelling
        try:
            assert num_points % 2 == 1
        except AssertionError:
            raise ValueError("Value for num_points must be odd")

        self.num_points = num_points

        self.dx = 2*x_range / (self.num_points - 1)
        _x = np.arange(self.dx, x_range + self.dx, self.dx)
        self.x = np.r_[-_x[::-1], 0, _x]
        self.psf = psf

        self.r1 = Parameter(name='r1', value=5, min=2, max=10)
        self.a1 = Parameter(name='a1', value=0.5, min=0)
        self.r2 = Parameter(name='r2', value=5, min=2, max=10)
        self.a2 = Parameter(name='a2', value=0.5, min=0)

    def __call__(self, x, **kwargs):
        r1 = kwargs.pop('r1', self.r1.value)
        r2 = kwargs.pop('r2', self.r2.value)
        a1 = kwargs.pop('a1', self.a1.value)
        a2 = kwargs.pop('a2', self.a2.value)

        return a1*self.get_y1(x, r1) + a2*self.get_y2(x, r2)

    @property
    def kernel_x(self):
        #todo ref Gaussian approximations of fluorescence microscope point-spread function models Bo Zhang, Josiane Zerubia, and Jean-Christophe Olivo-Marin
        x_range = 3*self.psf.sigma

        _x = np.arange(self.dx, x_range + self.dx, self.dx)
        psf_x = np.r_[-_x[::-1], 0, _x]
        return psf_x

    def signal_membrane(self, r):
        sqrt_arg = 1 + (self.x ** 2 / (r ** 2 - self.x ** 2))
        sqrt_arg[sqrt_arg < 0] = 0
        yvals = np.sqrt(sqrt_arg)

        conv = self._convolute(yvals)
        conv /= conv.max()
        return self.x, conv

    def signal_cytosol(self, r):
        # https://www.math.univ-toulouse.fr/~capitain/EJP.pdf
        sqrt_arg = r**2 - self.x**2
        sqrt_arg[sqrt_arg < 0] = 0
        yvals = 2*np.sqrt(sqrt_arg)

        conv = self._convolute(yvals)
        conv /= conv.max()
        return self.x, conv

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

    def get_y1(self, x, r):
        x_, y_ = self.signal_cytosol(r)
        return np.interp(x, x_, y_)

    def get_y2(self, x, r):
        x_, y_ = self.signal_membrane(r)
        return np.interp(x, x_, y_)

    def get_y(self, x, res_dict):
        #deprecate! in favour of __call__
        a1 = res_dict.get('a1', 0)
        a2 = res_dict.get('a2', 0)

        r1 = res_dict.get('r1', self.r1.value)
        r2 = res_dict.get('r2', self.r2.value)

        y1 = np.zeros_like(x) if a1 == 0 else a1*self.get_y1(x, r1)
        y2 = np.zeros_like(x) if a2 == 0 else a2*self.get_y2(x, r2)

        return y1 + y2

    def sub_par(self, res_dict):
        for k, v in res_dict.items():
            getattr(self, k).value = v

    def _convolute(self, arr):
        psf_y = self.psf(self.kernel_x)
        i = int((len(psf_y) / 2))

        res = np.convolve(arr, psf_y)
        return res[i:-i]