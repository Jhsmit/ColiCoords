import numpy as np
from colicoords.optimizers import Parameter
from scipy.integrate import quad
from joblib import Memory

#todo generator for allowing to load in experimental data to
class PSF(object):
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, x):
        return (1/self.sigma*np.sqrt(2*np.pi)) * np.exp(-(x/self.sigma)**2 / 2)

def sq_function(x):
    print('this is sq func')
    return x**2


def _y1(x, psf, r1):
    print('new y2', r1)

    def integrant(x, v, r1, psf):
        return psf(x - v) * np.nan_to_num(np.sqrt(r1 ** 2 - x ** 2))

    yarr, yerr = np.array([quad(integrant, -np.inf, np.inf, args=(v, r1, psf)) for v in x]).T

    return yarr


def _y2(x, psf, r2):
    print('new y2', r2)

    def integrant(x, v, r2, psf):
        try:
            return psf(x - v) * np.nan_to_num(np.sqrt(1 + (x ** 2 / (r2 ** 2 - x ** 2))))
        except ZeroDivisionError:
            return 0

        # return psf(x - v) * np.nan_to_num(np.sqrt(1 + (x ** 2 / (r2 ** 2 - x ** 2))))

    yarr, yerr = np.array([quad(integrant, -np.inf, np.inf, args=(v, r2, psf)) for v in x]).T

    return yarr


class RDistModelMem(object):

    def __init__(self, psf, mem=None):  #todo more options for psf modelling
        self.psf = psf

        self.r1 = Parameter(name='r1', value=4.5, min=2, max=6)
        self.a1 = Parameter(name='a1', value=0.5, min=0)
        self.r2 = Parameter(name='r2', value=5.5, min=2, max=8)
        self.a2 = Parameter(name='a2', value=0.5, min=0)

        self.yerr = None

        if mem is not None:
            self.y1 = mem.cache(_y1)
            self.y2 = mem.cache(_y2)
        else:
            self.y1 = self._y1
            self.y2 = self._y2

        self.i = 100

    def __call__(self, x, **kwargs):
        r1 = kwargs.pop('r1', self.r1.value)
        r2 = kwargs.pop('r2', self.r2.value)
        a1 = kwargs.pop('a1', self.a1.value)
        a2 = kwargs.pop('a2', self.a2.value)

        #a2 = 1 - a1

        if self.i:
            if r1 == 0:
                print("wat'")
           # print('r1', r1)


            r1_l = int(np.floor(self.i * r1)) / self.i
            r1_u = int(np.ceil(self.i * r1)) / self.i

            if r1_l == r1 and r1_u == r1:
                y1 = self.y1(x, self.psf, r1)


            else:
                y1_l = self.y1(x, self.psf, r1_l)
                y1_u = self.y1(x, self.psf, r1_u)

                y1 = ( y1_l*(r1_u - r1) + y1_u*(r1 - r1_l) ) / (r1_u - r1_l)



            r2_l = int(np.floor(self.i * r2)) / self.i
            r2_u = int(np.ceil(self.i * r2)) / self.i

            if r2_l == r2 and r2_u == r2:
                y2 = self.y2(x, self.psf, r2)
            else:

                y2_l = self.y2(x, self.psf, r2_l)
                y2_u = self.y2(x, self.psf, r2_u)

                y2 = (y2_l * (r2_u - r2) + y2_u * (r2 - r2_l)) / (r2_u - r2_l)
            #
            # r1 = int(round(self.i * r1)) / self.i
            # r2 = int(round(self.i * r2)) / self.i

        else:
            y1 = self.y1(x, self.psf, r1)
            y2 = self.y2(x, self.psf, r2)

        yarr = (a1 / (0.5 * np.pi * r1 ** 2))*y1 + (a2 / (np.pi * r2))*y2
        #



#        yarr = (a1 / (0.5 * np.pi * r1 ** 2))*self.y1(x, self.psf, r1) + (a2 / (np.pi * r2))*self.y2(x, self.psf, r2)
        yarr /= 2*np.pi


        return yarr

    @staticmethod
    def _y1(x, psf, r1):
        print('no mem', r1)
        def integrant(x, v, r1, psf):
            return psf(x - v) * np.nan_to_num(np.sqrt(r1 ** 2 - x ** 2))

        yarr, yerr = np.array([quad(integrant, -np.inf, np.inf, args=(v, r1, psf)) for v in x]).T

        return yarr

    @staticmethod
    def _y2(x, psf, r2):
        print('no mem', r2)
        def integrant(x, v, r2, psf):
            try:
                return psf(x - v) * np.nan_to_num(np.sqrt(1 + (x ** 2 / (r2 ** 2 - x ** 2))))
            except ZeroDivisionError:
                return 0

            #return psf(x - v) * np.nan_to_num(np.sqrt(1 + (x ** 2 / (r2 ** 2 - x ** 2))))

        yarr, yerr = np.array([quad(integrant, -np.inf, np.inf, args=(v, r2, psf)) for v in x]).T

        return yarr

    @staticmethod
    def integrant(x, v, psf, a1, a2, r1, r2):
        #normalization
        a1 /= 0.5 * np.pi * r1 ** 2
        a2 /= np.pi * r2
        #print(psf)
        # print(self.y1(x, a1, r1))
        # print(self.y2(x, a2, r2))
        # print(psf(x-v))

        res = psf(x-v) * ((a1 / (0.5 * np.pi * r1 ** 2))*np.nan_to_num(np.sqrt(r1 ** 2 - x ** 2)) +
                          (a2 / (np.pi * r2)) * np.nan_to_num(np.sqrt(1 + (x ** 2 / (r2 ** 2 - x ** 2)))))
        return res



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


class RDistModel(object):

    def __init__(self, psf):  #todo more options for psf modelling
        self.psf = psf

        self.r1 = Parameter(name='r1', value=5, min=2, max=10)
        self.a1 = Parameter(name='a1', value=0.5, min=0)
        self.r2 = Parameter(name='r2', value=5, min=2, max=10)
        self.a2 = Parameter(name='a2', value=0.5, min=0)

        self.yerr = None

    def __call__(self, x, **kwargs):
        r1 = kwargs.pop('r1', self.r1.value)
        r2 = kwargs.pop('r2', self.r2.value)
        a1 = kwargs.pop('a1', self.a1.value)
        a2 = kwargs.pop('a2', self.a2.value)

        yarr, self.yerr = np.array([quad(self.integrant, -np.inf, np.inf, args=(v, self.psf, a1, a2, r1, r2)) for v in x]).T

        return yarr

    @staticmethod
    def integrant(x, v, psf, a1, a2, r1, r2):
        #normalization
        a1 /= 0.5 * np.pi * r1 ** 2
        a2 /= np.pi * r2
        #print(psf)
        # print(self.y1(x, a1, r1))
        # print(self.y2(x, a2, r2))
        # print(psf(x-v))

        res = psf(x-v) * ((a1 / (0.5 * np.pi * r1 ** 2))*np.nan_to_num(np.sqrt(r1 ** 2 - x ** 2)) +
                          (a2 / (np.pi * r2)) * np.nan_to_num(np.sqrt(1 + (x ** 2 / (r2 ** 2 - x ** 2)))))
        return res

    @staticmethod
    def y1(x, a1, r1):
        return (a1 / (0.5 * np.pi * r1 ** 2))*np.nan_to_num(np.sqrt(r1 ** 2 - x ** 2))

    @staticmethod
    def y2(x, a2, r2):
        return (a2 / (np.pi * r2))*np.nan_to_num(np.sqrt(1 + (x ** 2 / (r2 ** 2 - x ** 2))))

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
