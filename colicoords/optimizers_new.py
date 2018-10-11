import numpy as np
from scipy.optimize import minimize, basinhopping, differential_evolution
from colicoords.config import cfg
from functools import partial
from abc import ABCMeta, abstractmethod
from symfit import Fit
from symfit.core.objectives import BaseObjective
from colicoords.models import CellModel, NumericalCellModel
from colicoords.minimizers import Powell, BaseMinimizer, NelderMead, BFGS, SLSQP, LBFGSB
from symfit.core.minimizers import BaseMinimizer
from symfit.core.fit import CallableModel


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

# def calc_binary_img(cell_obj, parameters):
#     cell_obj.coords.sub_par(parameters)
#     binary = cell_obj.coords.rc < cell_obj.coords
#     return binary.astype(int).flatten()
#
# def calc_img



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


class LinearModelFit(BaseFit):
    """Fitting of a linear model with two components.

    Apart from the linear coefficients the model can have (global) fit parameters. When fitting many datasets the
    amplitudes are return as an array while the other fit parameters are global.

        Attributes:
            val (:obj:`float`): Current chi-squared value.

    """
    def __init__(self, model, x, y):
        """

        Args:
            model :
            x (:class:`~numpy.ndarray`:): Array of x datapoints
            y_arr (:class:`~numpy.ndarray`:): Array of y datapoints. Either 1D of equal size to x or NxM with N datasets
                of length M equal to length of x.

        """
        self._model = model
        self._x = x
        self._y = y
        super(LinearModelFit, self).__init__()

    @property
    def objective(self):
        def _objective(par_values, par_names, model, x, y):
            par_dict = {par_name: par_value for par_name, par_value in zip(par_names, par_values)}

            y1 = model(x, **{'a1': 1, 'a2': 0}, **par_dict)
            y2 = model(x, **{'a1': 0, 'a2': 1}, **par_dict)

            a1, a2 = solve_linear_system(y1, y2, y)
            F = np.outer(a1, y1) + np.outer(a2, y2)

            return np.sum(F*(F - y)**2)
        return _objective

    def fit_amplitudes(self):
        """ Minimizes chi squared by finding amplitudes a1 and a2 in a1*y1 + a2*y2 == y.

        Returns:
            :obj:`tuple`: Tuple of amplitudes (a1, a2) solution

        """

        y1 = self.model(self.x, a1=1, a2=0)
        y2 = self.model(self.x, a1=0, a2=1)

        a1, a2 = solve_linear_system(y1, y2, self.y)
        return a1, a2

    def fit_parameters(self, parameters, bounds=None, constraint=True, solver='DE', solver_kwargs=None, **kwargs):
        """ Fit the current model and data optimizing given (global) *parameters*.

        Args:
            parameters (:obj:`str`): Parameters to fit. Format is a single string where parameters are separated by a space
            bounds: If *True* the model's :meth:`get_bounds` is called to determine the bounds. Otherwise, specify a sequence or
                :class:`scipy.optimize.Bounds` class as specified in :meth:`scipy.optimize.minimize`.
            constraint: If *True* the model's :meth:`get_constraints` is called to set constraints.
            solver (:obj:`str`): Either 'DE', 'basin_hop' or 'normal' to use :meth:scipy.optimize.differential_evolution`,
                :meth:scipy.optimize.basin_hop` or :meth:scip.optimize.minimize:, respectively.
            solver_kwargs: Optional kwargs to pass to the solver when using either differential evolution or basin_hop.
            **kwargs: Optional kwargs to pass to :meth:`scipy.optimize.minimize`.

        Returns:
            :`obj`:dict: Dictionary with fitting results. The entries are the global fit parameters as well as the amplitudes.
        """

        res_dict, val = super(LinearModelFit, self).fit_parameters(parameters, bounds=bounds, constraint=constraint,
                                                   solver=solver, solver_kwargs=solver_kwargs, **kwargs)

        y1 = self.model(self.x, **{'a1': 1, 'a2': 0}, **res_dict)
        y2 = self.model(self.x, **{'a1': 0, 'a2': 1}, **res_dict)

        res_dict['a1'], res_dict['a2'] = solve_linear_system(y1, y2, self.y)
        return res_dict, val

    def execute(self, bounds=None, constraint=None, solver='DE'):
        par = set(self.model.parameters.split(' '))
        linear_par = set(self.model.linear_parameters.split(' '))
        parameters = ' '.join(list(par - linear_par))

        res, v = self.fit_parameters(parameters, bounds=bounds, constraint=constraint, solver=solver)
        return res, v


class AbstractFit(BaseFit):
    """General class for fitting of a model with a set of parameters to a dataset with x- and y data"""

    def __init__(self, model, x, y):
        super(AbstractFit, self).__init__()
        self._model = model
        self._x = x
        self._y = y

    @property
    def objective(self):
        def _objective(par_values, par_names, model, x, y):
            par_dict = {par_name: par_value for par_name, par_value in zip(par_names, par_values)}
            y_model = model(x, **par_dict)

            yn = y / y.max()

            return np.sum(yn*((y - y_model)**2))
        return _objective


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





class set_params:
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


# might want to wrap the objective functions since they have the parameter substituion part in common
# def _wrap_func(par_values, par_names, cell_obj, data_name, objective, obj_kwargs):
#     par_dict = {par_name: par_value for par_name, par_value in zip(par_names, par_values)}
#     cell_obj.coords.sub_par(par_dict)
#     return objective(cell_obj, data_name, **obj_kwargs)


# Functions to minimize functions by matrix-fu
def solve_linear_system(y1, y2, data):
    """Solve system of linear eqns a1*y1 + a2*y2 == data but then also vector edition of that"""
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