import numpy as np
from scipy.optimize import minimize, basinhopping, differential_evolution
from colicoords.config import cfg
from colicoords.support import ArrayFitResults
from functools import partial
from abc import ABCMeta, abstractmethod
from symfit import Fit
from symfit.core.objectives import BaseObjective
from colicoords.models import CellModel, NumericalCellModel
from colicoords.minimizers import Powell, BaseMinimizer, NelderMead, BFGS, SLSQP, LBFGSB
from symfit.core.minimizers import BaseMinimizer
from symfit.core.fit import CallableNumericalModel, TakesData


class CellBaseObjective(BaseObjective):
    def __init__(self, cell_obj, data_name, *args, **kwargs):
        super(BaseObjective, self).__init__(*args, **kwargs)
        print('cellobj in base objective', cell_obj)

        self.cell_obj = cell_obj
        self.data_name = data_name


class NumericalBinaryXORObjective(CellBaseObjective):
    def __call__(self, **parameters):
        self.cell_obj.coords.sub_par(parameters)
        binary = self.cell_obj.coords.rc < self.cell_obj.coords.r
        return binary.astype(int)


class NumericalSimulatedCellObjective(CellBaseObjective):
    def __call__(self, **parameters):
        r = parameters.pop('r', self.cell_obj.coords.r)
        r = r / self.cell_obj.coords.r

        self.cell_obj.coords.sub_par(parameters)
        #todo check and make sure that the r_dist isnt calculated to far out which can give some strange results

        stop = np.max(self.cell_obj.data.shape) / 2
        step = 1

        #todo some way to access these kwargs
        xp, fp = self.cell_obj.r_dist(stop, step, data_name=self.data_name, method='box')
        simulated = np.interp(r * self.cell_obj.coords.rc, xp, np.nan_to_num(fp))  # todo check nantonum cruciality

        return simulated


class NumericalSTORMMembraneObjective(CellBaseObjective):
    def __init__(self, *args, **kwargs):
        self.r_upper = kwargs.pop('r_upper', None)
        self.r_lower = kwargs.pop('r_lower', lambda x: 2*np.std(x))
        super(STORMMembraneObjective, self).__init__(*args, **kwargs)

    def __call__(self, parameters):

        self.cell_obj.coords.sub_par(parameters)
        storm_data = self.cell_obj.data.data_dict[self.data_name]
        r_vals = self.cell_obj.coords.calc_rc(storm_data['x'], storm_data['y'])

        b_upper = r_vals < (r_vals.mean() + self.r_upper(r_vals)) if self.r_upper else True
        b_lower = r_vals > (r_vals.mean() - self.r_lower(r_vals)) if self.r_lower else True

        b = np.logical_and(b_upper, b_lower)

        r_vals = r_vals[b]
        return r_vals


class BinaryXORObjective(CellBaseObjective):
    def __call__(self, **parameters):
        self.cell_obj.coords.sub_par(parameters)
        binary = self.cell_obj.coords.rc < self.cell_obj.coords.r

        # todo squared!?

        return np.sum(np.logical_xor(self.cell_obj.data.data_dict[self.data_name], binary))


class SimulatedCellObjective(CellBaseObjective):
    def __call__(self, **parameters):
        r = parameters.pop('r', self.cell_obj.coords.r)
        r = r / self.cell_obj.coords.r

        self.cell_obj.coords.sub_par(parameters)
        #todo check and make sure that the r_dist isnt calculated to far out which can give some strange results

        stop = np.max(self.cell_obj.data.shape) / 2
        step = 1

        #todo some way to access these kwargs
        xp, fp = self.cell_obj.r_dist(stop, step, data_name=self.data_name, method='box')
        simulated = np.interp(r * self.cell_obj.coords.rc, xp, np.nan_to_num(fp))  # todo check nantonum cruciality

        real = self.cell_obj.data.data_dict[self.data_name]

        return np.sum((simulated - real)**2)


class STORMMembraneObjective(CellBaseObjective):
    def __init__(self, *args, **kwargs):
        self.r_upper = kwargs.pop('r_upper', None)
        self.r_lower = kwargs.pop('r_lower', lambda x: 2*np.std(x))
        super(STORMMembraneObjective, self).__init__(*args, **kwargs)

    def __call__(self, parameters):

        self.cell_obj.coords.sub_par(parameters)
        storm_data = self.cell_obj.data.data_dict[self.data_name]
        r_vals = self.cell_obj.coords.calc_rc(storm_data['x'], storm_data['y'])

        b_upper = r_vals < (r_vals.mean() + self.r_upper(r_vals)) if self.r_upper else True
        b_lower = r_vals > (r_vals.mean() - self.r_lower(r_vals)) if self.r_lower else True

        b = np.logical_and(b_upper, b_lower)

        r_vals = r_vals[b]
        #todo return inf when no valus are present anymore
        return np.sum(np.square(r_vals - self.cell_obj.coords.r))**2


class STORMAreaObjective(CellBaseObjective):
    def __init__(self, *args, **kwargs):
        self.maximize = kwargs.pop('maximize', 'localizations')
        super(STORMAreaObjective, self).__init__(*args, **kwargs)

    def __call__(self, parameters):
        self.cell_obj.coords.sub_par(parameters)
        storm_data = self.cell_obj.data.data_dict[self.data_name]
        r_vals = self.cell_obj.coords.calc_rc(storm_data['x'], storm_data['y'])
        bools = r_vals < self.cell_obj.coords.r

        if self.maximize == 'localizations':
            p = np.sum(bools)
        elif self.maximize == 'photons':
            p = np.sum(storm_data['intensity'][bools])
        else:
            raise ValueError('Invalid maximize keyword value')

        return -p / self.cell_obj.area


class CellFit(object):
    defaults = {
        'binary': NumericalBinaryXORObjective,
        'storm': NumericalSTORMMembraneObjective,
        'brightfield': NumericalSimulatedCellObjective,
        'fluorescence': NumericalSimulatedCellObjective,
    }

    def __init__(self, cell_obj, data_name='binary', objective=None, minimizer=Powell, **kwargs):
        self.cell_obj = cell_obj
        self.data_name = data_name
        self.minimizer = minimizer
        self.kwargs = kwargs

        dclass = self.data_elem.dclass
        obj = self.defaults[dclass] if not objective else objective

        obj(self.cell_obj, data_name)
        self.model = NumericalCellModel(cell_obj, obj(self.cell_obj, data_name))
        self.fit = Fit(self.model, self.data_elem, minimizer=minimizer, **kwargs)

    def renew_fit(self):
        self.fit = Fit(self.model, self.data_elem, minimizer=self.minimizer, **self.kwargs)

    def execute(self, **kwargs):
        return self.fit.execute(**kwargs)

    def fit_parameters(self, parameters, **kwargs):
        with set_params(self.fit, parameters):
            self.renew_fit()
            res = self.execute(**kwargs)
            for k, v in res.params.items():
                i = [par.name for par in self.model.params].index(k)
                self.model.params[i].value = v

        self.model.cell_obj.coords.sub_par(res.params)
        return res

    def fit_stepwise(self, **kwargs):
        i = 0
        j = 0
        prev_val = 0

        imax = kwargs.get('imax', 3)
        jmax = kwargs.get('jmax', 5)

        assert imax > 0
        assert jmax > 0
        while i < imax and j < jmax:
            #todo checking and testng
            j += 1
            res = self.fit_parameters('r', **kwargs)
            res = self.fit_parameters('xl xr', **kwargs)
            res = self.fit_parameters('a0 a1 a2', **kwargs)
            print('Current minimize value: {}'.format(res.objective_value))
            if prev_val == res.objective_value:
                i += 1
            prev_val = res.objective_value

        return res

    @property
    def data_elem(self):
        return self.cell_obj.data.data_dict[self.data_name]


class BaseFit(metaclass=ABCMeta):
    """base object for fitting"""

    def __init__(self):
      #  super(BaseFit, self).__init__()
        self.val = None

    @property
    def objective(self):
        try:
            return self._objective
        except AttributeError:
            raise NotImplementedError("Subclasses of BaseFit must set the 'model' attribute")

    @property
    def model(self):
        try:
            return self._model
        except AttributeError:
            raise NotImplementedError("Subclasses of BaseFit must set the 'model' attribute")

    @property
    def x(self):
        try:
            return self._x
        except AttributeError:
            raise NotImplementedError("Subclasses of BaseFit must set the 'x' attribute")

    @property
    def y(self):
        try:
            return self._y
        except AttributeError:
            raise NotImplementedError("Subclasses of BaseFit must set the 'y' attribute")

    def fit(self, minimizer=None, minimize_options=None, **kwargs):
        return self.fit_parameters(self.model.params, minimizer=minimizer, minimize_options=minimize_options, **kwargs)

    def fit_parameters(self, parameters, minimizer=None, minimize_options=None, **kwargs):
        """ Fit the current model and data optimizing given (global) *parameters*.

        Args:
            parameters (:obj:`str`): Parameters to fit. Format is a single string where paramers are separated by a spaces.
            solver (:obj:`str`): Either 'DE', 'basin_hop' or 'normal' to use :meth:scipy.optimize.differential_evolution`,
                :meth:scipy.optimize.basin_hop` or :meth:scip.optimize.minimize:, respectively.
            solver_kwargs: Optional kwargs to pass to the solver when using either differential evolution or basin_hop.
            **kwargs: Optional kwargs to pass to :meth:`scipy.optimize.minimize`.

        Returns:
            :`obj`:dict: Dictionary with fitting results. The entries are the global fit parameters as well as the amplitudes.
        """

        #todo context manager for fixing / unfixing params
        params_list = [par.rstrip(',.: ') for par in parameters.split(' ')]
        original_fixing = [par.fixed for par in self.model.params]
        fixed_params = [par for par in self.model.params if not par.name in params_list]
        for par in fixed_params:
            par.fixed = True

        # print('----before----')
        # for par in self.model.params:
        #     print(par.name, par.fixed)
        # print('----before----')

        minimizer = Powell if not minimizer else minimizer
        assert issubclass(minimizer, BaseMinimizer)

        fit = Fit(self.model, objective=self.objective, minimizer=minimizer, **kwargs)
        minimize_options = minimize_options or {}

        res = fit.execute(**minimize_options)

        self.val = self.objective(**res.params)

        for par, fixed in zip(self.model.params, original_fixing):
            par.fixed = fixed

        # print('----after----')
        # for par in self.model.params:
        #     print(par.name, par.fixed)
        # print('----after----')

        #new values in model
        for k, v in res.params.items():
            i = [par.name for par in self.model.params].index(k)
            self.model.params[i].value = v

        self.model.cell_obj.coords.sub_par(res.params)

        return res.params, self.val


class CellOptimizer(BaseFit):
    """ Class for cell coordinate optimizing
    """

    defaults = {
        'binary': BinaryXORObjective,
        'storm': STORMMembraneObjective,
        'brightfield': SimulatedCellObjective,
        'fluorescence': SimulatedCellObjective,
    }

    def __init__(self, cell_obj, data_name='binary', objective=None):
        super(CellOptimizer, self).__init__()
        self.cell_obj = cell_obj
        self.data_name = data_name
        dclass = self.data_elem.dclass
        obj = self.defaults[dclass] if not objective else objective

        self._objective = obj(self.cell_obj, data_name)
        self._model = CellModel(self.cell_obj)
        par_dict = {par.name: par.value for par in self.model.params}
        start_val = self._objective(**par_dict)
        print('startval', start_val)

    def optimize(self, **kwargs):
        parameters = ' '.join([par.name for par in self.model.full_params])
        res, val = self.fit_parameters(parameters, **kwargs)
        self.val = val
        return res, val

    @property
    def data_elem(self):
        return self.cell_obj.data.data_dict[self.data_name]

    def optimize_stepwise(self, **kwargs):
        i = 0
        j = 0
        prev_val = 0

        imax = kwargs.get('imax', 3)
        jmax = kwargs.get('jmax', 5)

        assert imax > 0
        assert jmax > 0
        while i < imax and j < jmax:
            #todo checking and testng
            j += 1
            res, val = self.fit_parameters('r', **kwargs)
            res, val = self.fit_parameters('xl xr', **kwargs)
            res, val = self.fit_parameters('a0 a1 a2', **kwargs)
            print('Current minimize value: {}'.format(val))
            if prev_val == val:
                i += 1
            prev_val = val

        self.val = val
        return res, val

    def sub_par(self, par_dict):
        self.cell_obj.coords.sub_par(par_dict)



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
        linear_dict = {par.name: value for par, value in zip(self._old_model.linear_params, [func.a_list for func in self.model.numerical_components][0])}
        # î assuming all linear parameters are in all numerical components which is not true i guess
        overall_dict = {**res.params, **linear_dict}
        popt = [overall_dict[par.name] for par in self._old_model.params]

        return ArrayFitResults(self._old_model, popt, None, res.infodict, res.status_message, res.iterations, **res.gof_qualifiers)


def solve_linear_system(y_list, data):
    """Solve system of linear eqns a1*y1 + a2*y2 == data but then also vector edition of that"""
    D_vec = np.stack([data.dot(y) for y in y_list]).flatten()
    M = np.array([y1.dot(y2) for y1 in y_list for y2 in y_list]).reshape(len(y_list), len(y_list))

    l = len(data) if data.ndim == 2 else 1
    bigM = np.kron(M, np.eye(l))
    a1a2 = np.linalg.solve(bigM, D_vec)

    return np.split(a1a2, len(y_list))


class set_params(object):
    def __init__(self, fit_object, parameters):
        self.parametes = [par.rstrip(',.: ') for par in parameters.split(' ')]
        self.fit_object = fit_object

    def __enter__(self):
    # todo context manager for fixing / unfixing params
        self.original_fixing = [par.fixed for par in self.fit_object.model.params]
        fixed_params = [par for par in self.fit_object.model.params if not par.name in self.parametes]
        for par in fixed_params:
            par.fixed = True

    def __exit__(self, *args):
        for par, fixed in zip(self.fit_object.model.params, self.original_fixing):
            par.fixed = fixed


class wrapped_func(object):
    def __init__(self, func, data, linear_params):
        self.func = func # the original callable
        self.data = data # only data  no dict
        self.linear_params = linear_params  # list of names only

    def __call__(self, *args, **kwargs):
        y_list = [self.func(*args, **{par: 1 if par == s_par else 0 for par in self.linear_params}, **kwargs) for s_par in self.linear_params]
        self.a_list = solve_linear_system(y_list, self.data)
        result = sum([a_elem[:, np.newaxis] * y_elem for a_elem, y_elem in zip(self.a_list, y_list)])
        return result


def make_linear_model(model, data):
    new_dict = {k: wrapped_func(v, data[k], [par.name for par in model.linear_params]) for k, v in model.model_dict.items()}
    new_params = [par for par in model.params if par not in model.linear_params]
    return CallableNumericalModel(new_dict, model.independent_vars, new_params)