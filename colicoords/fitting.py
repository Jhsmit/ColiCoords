import numpy as np
import numbers
from colicoords.support import ArrayFitResults
from symfit import Fit
from colicoords.models import NumericalCellModel
from colicoords.minimizers import Powell
from symfit.core.fit import CallableNumericalModel, TakesData


class RadialData(np.lib.mixins.NDArrayOperatorsMixin):
    """
    Class mimicking a numpy ndarray used as dependent data for fitting STORM-membrane data.

    The apparent value of this object is an array with length `length` and whoes values are all equal to the radius of
    `cell_obj`.

    Parameters
    ----------
    cell_obj : :class:`~colicoords.cell.Cell`
        Cell object whos radius gives this array's values.
    length : :obj:`int`
        Length of the array.
    """

    _HANDLED_TYPES = (np.ndarray, numbers.Number)

    def __init__(self, cell_obj, length):
        self._cell_obj = cell_obj
        self._len = length
        self._array = np.ones(length, dtype=np.float)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        out = kwargs.get('out', ())
        for x in inputs + out:
            # Only support operations with instances of _HANDLED_TYPES.
            # Use ArrayLike instead of type(self) for isinstance to
            # allow subclasses that don't override __array_ufunc__ to
            # handle ArrayLike objects.
            if not isinstance(x, self._HANDLED_TYPES + (RadialData,)):
                return NotImplemented

        # Defer to the implementation of the ufunc on unwrapped values.
        inputs = tuple(x.value if isinstance(x, RadialData) else x
                       for x in inputs)
        if out:
            kwargs['out'] = tuple(
                x.value if isinstance(x, RadialData) else x
                for x in out)
        result = getattr(ufunc, method)(*inputs, **kwargs)

        if type(result) is tuple:
            # multiple return values
            return tuple(np.array(x) for x in result)
        elif method == 'at':
            # no return value
            return None
        else:
            # one return value
            return np.array(result)

    @property
    def shape(self):
        return self.value.shape

    @property
    def value(self):
        return self._cell_obj.radius*self._array


class CellMinimizeFunctionBase(object):
    """
    Base class for Objective objects used by ``CellFit`` to optimize the coordinate system.

    The base class takes a :class:`~colicoords.cell.Cell` object and the name of target data element to perform
    optimization on. Subclasses of ``CellMinimizeFunctionBase`` must implement the `__call__` builtin, which takes the
    coordinate system's parameters as keyword arguments.

    Note that this is not an objective function to be minimized, but instead the return value is compared with the
    specified data element or specific target data to give the chi-squared.

    Parameters
    ----------
    cell_obj : :class:`~colicoords.cell.Cell`
        Cell object to optimize.
    data_name : :obj:`str`
        Target data element name.
    """
    def __init__(self, cell_obj, data_name):
        self.cell_obj = cell_obj
        self.data_name = data_name


class CellBinaryFunction(CellMinimizeFunctionBase):
    """
    Binary data element objective function.

    Calling this object with coordinate system parameters returns a binary image by thresholding the radial distance
    image with the radius of the cell.
    """
    def __call__(self, **parameters):
        self.cell_obj.coords.sub_par(parameters)
        binary = self.cell_obj.coords.rc < self.cell_obj.coords.r
        return binary.astype(int)


class CellImageFunction(CellMinimizeFunctionBase):
    """
    Image element objective function.

    Calling this object with coordinate system parameters returns a reconstructed image of the target data element.
    """
    #todo add additional parameters to init parameters
    def __call__(self, **parameters):
        r = parameters.pop('r', self.cell_obj.coords.r)
        r = r / self.cell_obj.coords.r

        self.cell_obj.coords.sub_par(parameters)
        #todo check and make sure that the r_dist isnt calculated to far out which can give some strange results

        stop = np.max(self.cell_obj.data.shape) / 2
        step = 1

        #todo some way to access these kwargs
        #todo scipy.fftsmth.convolve
        xp, fp = self.cell_obj.r_dist(stop, step, data_name=self.data_name, method='box')
        simulated = np.interp(r * self.cell_obj.coords.rc, xp, np.nan_to_num(fp))  # todo check nantonum cruciality

        return simulated


class CellSTORMMembraneFunction(CellMinimizeFunctionBase):
    """
    STORM membrane objective function.

    Calling this object with coordinate system parameters returns a reconstructed image of the target data element.
    """
    #todo booleans is not going to work here, needs to be done via sigma_y!
    #todo remove init
    def __init__(self, *args, **kwargs):
        # self.r_upper = kwargs.pop('r_upper', None)
        # self.r_lower = kwargs.pop('r_lower', lambda x: 2*np.std(x))
        super(CellSTORMMembraneFunction, self).__init__(*args, **kwargs)

    def __call__(self, **parameters):
        self.cell_obj.coords.sub_par(parameters)
        storm_data = self.cell_obj.data.data_dict[self.data_name]
        r_vals = self.cell_obj.coords.calc_rc(storm_data['x'], storm_data['y'])
        # b_upper = r_vals < (r_vals.mean() + self.r_upper(r_vals)) if self.r_upper else True
        # b_lower = r_vals > (r_vals.mean() - self.r_lower(r_vals)) if self.r_lower else True
        #
        # b = np.logical_and(b_upper, b_lower)
        r_vals = r_vals
        return r_vals.astype(np.float)

    @property
    def target_data(self):
        """Dependent (target) data for coordinate optimization based on STORM membrane markers"""
        return RadialData(self.cell_obj, len(self.cell_obj.data.data_dict[self.data_name]['x']))


class DepCellFit(Fit):
    # in this implementation stepwise fitting doesnt work, however direct subclassing of Fit is preferred. Reasses with
    # new symfit version

    defaults = {
        'binary': CellBinaryFunction,
        'storm': CellSTORMMembraneFunction,
        'brightfield': CellImageFunction,
        'fluorescence': CellImageFunction,
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
        super(DepCellFit, self).__init__(self.model, self.data_elem, minimizer=minimizer, **kwargs)

    def renew_fit(self):
        super(DepCellFit, self).__init__(self.model, self.data_elem, minimizer=self.minimizer, **self.kwargs)

    def fit_parameters(self, parameters, **kwargs):
        with set_params(self, parameters):
            super(DepCellFit, self).__init__(self.model, self.data_elem, minimizer=self.minimizer, **self.kwargs)
            res = self.execute(**kwargs)
            for k, v in res.params.items():
                i = [par.name for par in self.model.params].index(k)
                self.model.params[i].value = v

        self.model.cell_obj.coords.sub_par(res.params)
        return res

    def execute_stepwise(self, **kwargs):
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


class CellFit(object):

    # Default functions to use for given data classes.
    defaults = {
        'binary': CellBinaryFunction,
        'storm': CellSTORMMembraneFunction,
        'brightfield': CellImageFunction,
        'fluorescence': CellImageFunction,
    }

    def __init__(self, cell_obj, data_name='binary', cell_function=None, minimizer=Powell, **kwargs):
        self.cell_obj = cell_obj
        self.data_name = data_name
        self.minimizer = minimizer
        self.kwargs = kwargs

        dclass = self.data_elem.dclass
        func_klass = self.defaults[dclass] if not cell_function else cell_function

        if issubclass(func_klass, CellMinimizeFunctionBase):
            self.cell_function = func_klass(self.cell_obj, data_name)
        elif callable(func_klass):
            self.cell_function = func_klass
        else:
            raise TypeError("Invalid type for cell_function keyword argument.")

        self.model = NumericalCellModel(cell_obj, self.cell_function)
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

    def execute_stepwise(self, **kwargs):
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
        try:
            return self.cell_function.target_data
        except AttributeError:
            return self.cell_obj.data.data_dict[self.data_name]


class LinearModelFit(Fit):
    """
    Fitting of a model with linear parameters where the linear parameters are not fitted by ``symfit`` but instead
    solved as a system of linear equations.

    Parameters
    ----------
    model : :
    """
    def __init__(self, model, *args, **kwargs):
        objective = kwargs.pop('objective', None)
        minimizer = kwargs.pop('minimizer', None)
        constraints = kwargs.pop('constraints', None)
        temp_data = TakesData(model, *args, **kwargs)
        self._old_model = model
        self._new_model = make_linear_model(model, temp_data.dependent_data)
        super(LinearModelFit, self).__init__(self._new_model, *args, **kwargs,
                                             minimizer=minimizer, objective=objective, constraints=constraints)

    def execute(self, **kwargs):
        res = super(LinearModelFit, self).execute(**kwargs)
        linear_dict = {par.name: value for par, value in
                       zip(self._old_model.linear_params, [func.a_list for func in self.model.numerical_components][0])}
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
    ai = np.linalg.solve(bigM, D_vec)

    return np.split(ai, len(y_list))


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
