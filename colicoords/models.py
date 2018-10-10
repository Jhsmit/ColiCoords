import numpy as np
from scipy.integrate import quad
from colicoords.config import cfg
from symfit.core.fit import CallableModel, CallableNumericalModel, TakesData
from functools import partial
from symfit import Parameter, Variable, Fit
from symfit.core.objectives import BaseObjective, LeastSquares
from symfit.core.support import key2str
from symfit.core.fit import FitResults
import copy
import inspect


class NumericalCellModel(CallableNumericalModel):
    def __init__(self, cell_obj, objective):
        self.cell_obj = cell_obj
        r = Parameter('r', value=cell_obj.coords.r, min=cell_obj.coords.r / 4, max=cell_obj.coords.r * 4)
        xl = Parameter('xl', value=cell_obj.coords.xl,
                            min=cell_obj.coords.xl - cfg.ENDCAP_RANGE / 2, max=cell_obj.coords.xl + cfg.ENDCAP_RANGE / 2)
        xr = Parameter('xr', value=cell_obj.coords.xr,
                            min=cell_obj.coords.xr - cfg.ENDCAP_RANGE / 2, max=cell_obj.coords.xr + cfg.ENDCAP_RANGE / 2)
        a0 = Parameter('a0', value=cell_obj.coords.coeff[0], min=0, max=cell_obj.data.shape[0]*1.5)
        a1 = Parameter('a1', value=cell_obj.coords.coeff[1], min=-15, max=15)
        a2 = Parameter('a2', value=cell_obj.coords.coeff[2], min=-0.05, max=0.05)

        y = Variable('y')

        parameters = [a0, a1, a2, r, xl, xr]
        super(NumericalCellModel, self).__init__({y: objective}, [], parameters)


class CustomCallableModel(CallableModel):
    def __init__(self, parameters, variables, *args, **kwargs):
        super(CustomCallableModel, self).__init__({})
        self.params = sorted(parameters, key=str)
        self.full_params = self.params.copy()
        self.dependent_vars = variables
        self.sigmas = {Variable('y'): Variable('sigma_y')}
        self.__signature__ = self._make_signature()
        self.model_dict = {Variable('y'): None}


class CellModel(CustomCallableModel):
    def __init__(self, cell_obj):
        self.cell_obj = cell_obj

        r = Parameter('r', value=cell_obj.coords.r, min=cell_obj.coords.r / 4, max=cell_obj.coords.r * 4)
        xl = Parameter('xl', value=cell_obj.coords.xl,
                            min=cell_obj.coords.xl - cfg.ENDCAP_RANGE / 2, max=cell_obj.coords.xl + cfg.ENDCAP_RANGE / 2)
        xr = Parameter('xr', value=cell_obj.coords.xr,
                            min=cell_obj.coords.xr - cfg.ENDCAP_RANGE / 2, max=cell_obj.coords.xr + cfg.ENDCAP_RANGE / 2)
        a0 = Parameter('a0', value=cell_obj.coords.coeff[0], min=0, max=cell_obj.data.shape[0]*1.5)
        a1 = Parameter('a1', value=cell_obj.coords.coeff[1], min=-15, max=15)
        a2 = Parameter('a2', value=cell_obj.coords.coeff[2], min=-0.05, max=0.05)

        y = Variable('y')

        parameters = [a0, a1, a2, r, xl, xr]
        variables = [y]

        super(CellModel, self).__init__(parameters, variables)

    def __str__(self):
        return 'cell_model'

    def eval_components(self, *args, **kwargs):
        return [0]


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


class wrapped_func(object):
    def __init__(self, func, data, linear_params):
        self.func = func # the original callable
        self.data = data # only data  no dict
        self.linear_params = linear_params  # list of names only

    def __call__(self, *args, **kwargs):
        y_list = [self.func(*args, **{par: 1 if par == s_par else 0 for par in self.linear_params}, **kwargs) for s_par in self.linear_params]
        self.a_list = solve_linear_system(y_list, self.data)

        result = sum([a_elem[:, np.newaxis] * y_elem for a_elem, y_elem in zip(self.a_list, y_list)])
        return result.flatten()

def make_linear_model(model, data):
    new_dict = {k: wrapped_func(v, data[k], [par.name for par in model.linear_params]) for k, v in model.model_dict.items()}
    new_params = [par for par in model.params if par not in model.linear_params]
    return CallableNumericalModel(new_dict, model.independent_vars, new_params)


class LinearModelFit(Fit):
    def __init__(self, model, *args, **kwargs):
        objective = kwargs.pop('objective', None)
        minimizer = kwargs.pop('minimizer', None)
        constraints = kwargs.pop('constraints', None)
        temp_data = TakesData(model, *args, **kwargs)
        self._old_model = model
        self._new_model = make_linear_model(model, temp_data.dependent_data)
        super(LinearModelFit, self).__init__(self._new_model, *args, **kwargs, minimizer=minimizer, objective=objective, constraints=constraints)

    def execute(self, **kwargs):
        res = super(LinearModelFit, self).execute(**kwargs)
        linear_dict = {par.name: float(value) for par, value in zip(self._old_model.linear_params, [func.a_list for func in self.model.numerical_components][0])}
        # î assuming all linear parameters are in all numerical components which is not true i guess
        overall_dict = {**res.params, **linear_dict}
        popt = [overall_dict[par.name] for par in self._old_model.params]

        return FitResults(self._old_model, popt, None, res.infodict, res.status_message, res.iterations, **res.gof_qualifiers)
    #
    #
    # def old__init__(self, model, *args, **kwargs):
    #
    #
    #
    #
    #     tempdata = TakesData(model, *args, **kwargs)
    #     print(model.params)
    #     self.original_objective = next(iter(model.model_dict.values()))
    #     self.nonlinear_objective = copy.deepcopy(LeastSquares(model, tempdata.data))
    #     print('call nonlinear obj', self.nonlinear_objective(x=10, a1=2, a2=0, r1=4, r2=20))
    #     __dontthouchthis = copy.deepcopy(self.nonlinear_objective)
    #
    #
    #     new_params = [par for par in model.params if par not in model.linear_params]
    #     print('new params', new_params)
    #     print('before super call', model.independent_vars)
    #     #print(self.model.__bases__)
    #     #super(CallableNumericalModel, model).__init__(model.model_dict, model.independent_vars, new_params)
    #     print('dict and vars')
    #     print(model.model_dict)
    #     print(model.independent_vars)
    #     print(model.dependent_vars)
    #
    #     objective = LinearEquationsObjective( self.original_objective, self.nonlinear_objective)
    #     new_dict = {model.dependent_vars[0]: objective}
    #     new_model = CallableNumericalModel(new_dict, model.independent_vars, new_params)
    #
    #     #self.model.__init__(self.model.model_dict, self.model.vars, new_params)
    #
    #     print('updated model params', new_model.params)
    #     print(*args)
    #     super(LinearModelFit, self).__init__(new_model, *args, minimizer=minimizer, constraints=constraints, objective=objective, **kwargs)
    #
    #
    #     print('call nonlinear obj_new',  self.nonlinear_objective(x=10, a1=2, a2=0, r1=4, r2=20))
    #     self.objective = LinearEquationsObjective( self.original_objective, self.nonlinear_objective)
    #     print(self.objective)
    #
    #     print('model params', self.model.params)
    #
    # def older___init__(self, *args, **kwargs):
    #     super(LinearModelFit, self).__init__(*args, **kwargs)
    #
    #
    #     new_params = [par for par in self.model.params if par not in self.model.linear_params]
    #     print('new params', new_params)
    #     print('before super call', self.model.independent_vars)
    #     #print(self.model.__bases__)
    #     super(CallableNumericalModel, self.model).__init__(self.model.model_dict, self.model.independent_vars, new_params)
    #     #self.model.__init__(self.model.model_dict, self.model.vars, new_params)
    #
    #     print('updated model params', self.model.params)
    #
    #     super(LinearModelFit, self).__init__(self.model, *args[1:], **kwargs)
    #
    #
    #     #self.objective = LinearEquationsObjective(self.nonlinear_objective)
    #
    #
    #     #print(self.objective)
    #
    #     print('model params', self.model.params)


def solve_linear_system(y_list, data):
    """Solve system of linear eqns a1*y1 + a2*y2 == data but then also vector edition of that"""
    y1, y2 = y_list
    Dy1 = data.dot(y1)
    Dy2 = data.dot(y2)

    D_vec = np.stack((Dy1, Dy2)).flatten()

    y1y1 = y1.dot(y1)
    y1y2 = y1.dot(y2)
    y2y2 = y2.dot(y2)

    M = np.array([[y1y1, y1y2], [y1y2, y2y2]])
    l = len(data) if data.ndim == 2 else 1
    bigM = np.kron(M, np.eye(l))
    a1a2 = np.linalg.solve(bigM, D_vec)

    return np.split(a1a2, 2)


class RDistModel(CallableNumericalModel):
    def __init__(self, psf, mem=None):
        self.r1 = Parameter(name='r1', value=4.5, min=2, max=6)
        self.a1 = Parameter(name='a1', value=0.5, min=0)
        self.r2 = Parameter(name='r2', value=5.5, min=2, max=8)
        self.a2 = Parameter(name='a2', value=0.5, min=0)

        self.x = Variable('x')
        self.y = Variable('y')

        func = RDistObjective(psf, mem)
        parameters = [self.a1, self.a2, self.r1, self.r2]
        self.linear_params = [self.a1, self.a2]
        super(RDistModel, self).__init__({self.y: func}, [self.x], parameters)


class RDistObjective(object):
    def __init__(self, psf, mem=None):
        self.psf = psf

        if mem is not None:
            self.y1 = mem.cache(_y1, ignore=['psf'])
            self.y2 = mem.cache(_y2, ignore=['psf'])
        else:
            self.y1 = _y1
            self.y2 = _y2

        self.i = 10

    def __call__(self, x, **kwargs):
        r1 = kwargs.pop('r1')
        r2 = kwargs.pop('r2')
        a1 = kwargs.pop('a1')
        a2 = kwargs.pop('a2')

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
