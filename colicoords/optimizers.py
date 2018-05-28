import numpy as np
from scipy.optimize import minimize, basinhopping, differential_evolution
from colicoords.config import cfg
from functools import partial


class Parameter(object):
    def __init__(self, name, value=1., min=1.e-10, max=None):
        """

        Args:
            name (:obj:`str`): Name of the parameter.
            value (:obj:`float`): Initial guess value.
            min (:obj:`float`): Minimum bound. Use *None* for unbounded.
            max (:obj:`float`): Maximum bound. use *None* for unbounded.
        """
        self.name = name
        self.min = min
        self.max = max
        self.value = value


class BaseFit(object):
    """base object for fitting"""

    def __init__(self):
        self.val = None

    @property
    def objective(self):
        raise NotImplementedError()

    @property
    def model(self):
        raise NotImplementedError()

    @property
    def x(self):
        raise NotImplementedError()

    @property
    def y(self):
        raise NotImplementedError()

    def fit_parameters(self, parameters, bounds=None, constraint=True, solver='DE', solver_kwargs=None, **kwargs):
        """ Fit the current model and data optimizing given (global) *parameters*.

        Args:
            parameters (:obj:`str`): Parameters to fit. Format is a single string where paramers are separated by a space
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

        bounds = self.model.get_bounds(parameters, parameters if type(bounds) == bool else bounds) if bounds else None
        method = kwargs.pop('method', None)
        verbose = kwargs.pop('verbose', False)
        par_values = np.array([getattr(self.model, par).value for par in parameters.split(' ')])
        constraints = self.model.get_constraints(parameters) if constraint else None

        solver_kwargs = {} if solver_kwargs is None else solver_kwargs

        # todo this needs some checking if not all parameters are bounded
        accept_test = partial(self._accept_test, bounds)

        # todo use mystic for constraints: https://github.com/uqfoundation/mystic/blob/master/mystic/differential_evolution.py
        if solver == 'DE':
            result = differential_evolution(self.objective, bounds,
                                            args=(parameters.split(' '), self.model, self.x, self.y_arr),
                                            **solver_kwargs
                                            )
        elif solver == 'basin_hop':
            stepsize = solver_kwargs.pop('stepsize', 1)
            niter = solver_kwargs.pop('niter', 200)
            result = basinhopping(self.objective, par_values,
                                  minimizer_kwargs={
                                      'args': (parameters.split(' '), self.model, self.x, self.y_arr),
                                      'bounds': bounds,
                                      'method': method,
                                      'constraints': constraints,
                                      'options': {'disp': verbose},
                                      **kwargs},
                                  stepsize=stepsize,
                                  niter=niter,
                                  accept_test=accept_test,
                                  **solver_kwargs
                                  )
        elif solver == 'normal':
            result = minimize(self.objective, par_values, args=(parameters.split(' '), self.model, self.x, self.y_arr),
                              bounds=bounds, method=method, constraints=constraints, options={'disp': verbose},
                              **kwargs)
        else:
            raise ValueError("Value for 'solver' must be either 'DE', 'basin_hop' or 'normal'")

        try:
            res_dict = {key: val for key, val in zip(parameters.split(' '), result.x)}
        except TypeError:
            res_dict = {key: val for key, val in zip(parameters.split(' '), [result.x])}

        self.val = result.fun
        return res_dict, result.fun

    @staticmethod
    def _accept_test(bounds, **kwargs):
        par_values = kwargs['x_new']
        bools = [(-np.inf if pmin is None else pmin) <= val <= (np.inf if pmax is None else pmax)
                 for (pmin, pmax), val in zip(bounds, par_values)]

        return np.all(bools)


class LinearModelFit(object):
    """Fitting of a linear model with two components.

    Apart from the linear coefficients the model can have (global) fit parameters. When fitting many datasets the
    amplitudes are return as an array while the other fit parameters are global.

        Attributes:
            val (:obj:`float`): Current chi-squared value.

    """
    def __init__(self, model, x, y_arr):
        """

        Args:
            model :
            x (:class:`~numpy.ndarray`:): Array of x datapoints
            y_arr (:class:`~numpy.ndarray`:): Array of y datapoints. Either 1D of equal size to x or NxM with N datasets
                of length M equal to length of x.

        """
        self.model = model
        self.x = x
        self.y_arr = y_arr

        self.val = None

    def fit_amplitudes(self):
        """ Minimizes chi squared by finding amplitudes a1 and a2 in a1*y1 + a2*y2 == y.

        Returns:
            :obj:`tuple`: Tuple of amplitudus (a1, a2) solution

        """

        y1 = self.model(self.x, a1=1, a2=0)
        y2 = self.model(self.x, a1=0, a2=1)

        a1, a2 = solve_linear_system(y1, y2, self.y_arr)
        return a1, a2

    def fit_parameters(self, parameters, bounds=None, constraint=True, solver='DE', solver_kwargs=None, **kwargs):
        """ Fit the current model and data optimizing given (global) *parameters*.

        Args:
            parameters (:obj:`str`): Parameters to fit. Format is a single string where paramers are separated by a space
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
        def objective(par_values, par_names, model, x, y):
            par_dict = {par_name: par_value for par_name, par_value in zip(par_names, par_values)}

            y1 = model(x, **{'a1': 1, 'a2': 0}, **par_dict)
            y2 = model(x, **{'a1': 0, 'a2': 1}, **par_dict)

            a1, a2 = solve_linear_system(y1, y2, y)
            F = np.outer(a1, y1) + np.outer(a2, y2)

            return np.sum((F - y)**2)

        bounds = self.model.get_bounds(parameters, parameters if type(bounds) == bool else bounds) if bounds else None
        method = kwargs.pop('method', None)
        verbose = kwargs.pop('verbose', False)
        par_values = np.array([getattr(self.model, par).value for par in parameters.split(' ')])
        constraints = self.model.get_constraints(parameters) if constraint else None

        solver_kwargs = {} if solver_kwargs is None else solver_kwargs

        #todo this needs some checking if not all parameters are bounded
        def _accept_test(bounds, **kwargs):
            par_values = kwargs['x_new']
            bools = [(-np.inf if pmin is None else pmin) <= val <= (np.inf if pmax is None else pmax)
                     for (pmin, pmax), val in zip(bounds, par_values)]

            return np.all(bools)

        accept_test = partial(_accept_test, bounds)

        #todo use mystic for constraints: https://github.com/uqfoundation/mystic/blob/master/mystic/differential_evolution.py
        if solver == 'DE':
            result = differential_evolution(objective, bounds,
                                            args=(parameters.split(' '), self.model, self.x, self.y_arr),
                                            **solver_kwargs
                                            )
        elif solver == 'basin_hop':
            stepsize = solver_kwargs.pop('stepsize', 1)
            niter = solver_kwargs.pop('niter', 200)
            result = basinhopping(objective, par_values,
                                  minimizer_kwargs={
                                      'args': (parameters.split(' '), self.model, self.x, self.y_arr),
                                      'bounds': bounds,
                                      'method': method,
                                      'constraints': constraints,
                                      'options': {'disp': verbose},
                                      **kwargs},
                                  stepsize=stepsize,
                                  niter=niter,
                                  accept_test=accept_test,
                                  **solver_kwargs
                                  )
        elif solver == 'normal':
            result = minimize(objective, par_values, args=(parameters.split(' '), self.model, self.x, self.y_arr),
                              bounds=bounds, method=method, constraints=constraints, options={'disp': verbose}, **kwargs)
        else:
            raise ValueError("Value for 'solver' must be either 'DE', 'basin_hop' or 'normal'")

        try:
            res_dict = {key: val for key, val in zip(parameters.split(' '), result.x)}
        except TypeError:
            res_dict = {key: val for key, val in zip(parameters.split(' '), [result.x])}

        y1 = self.model(self.x, **{'a1': 1, 'a2': 0}, **res_dict)
        y2 = self.model(self.x, **{'a1': 0, 'a2': 1}, **res_dict)

        res_dict['a1'], res_dict['a2'] = solve_linear_system(y1, y2, self.y_arr)

        self.val = result.fun
        return res_dict, result.fun

    def _fit_global_de(self, parameters, bounds=True, constraint=True, **kwargs):
        # todo generalize to fit global vars and local vars
        def objective(par_values, par_names, model, x, y):
            par_dict = {par_name: par_value for par_name, par_value in zip(par_names, par_values)}

            y1 = model(x, **{'a1': 1, 'a2': 0}, **par_dict)
            y2 = model(x, **{'a1': 0, 'a2': 1}, **par_dict)

            a1, a2 = solve_linear_system(y1, y2, y)
            F = np.outer(a1, y1) + np.outer(a2, y2)

            return np.sum((F - y)**2)

        bounds = self.model.get_bounds(parameters, parameters if type(bounds) == bool else bounds) if bounds else None
        method = None
        verbose = kwargs.pop('verbose', False)
        par_values = np.array([getattr(self.model, par).value for par in parameters.split(' ')])
        constraints = self.model.get_constraints(parameters) if constraint else None

        result = differential_evolution(objective, bounds,
                                        args=(parameters.split(' '), self.model, self.x, self.y_arr)

                              )

        try:
            res_dict = {key: val for key, val in zip(parameters.split(' '), result.x)}
        except TypeError:
            res_dict = {key: val for key, val in zip(parameters.split(' '), [result.x])}

        self.val = result.fun
        return res_dict, result.fun


class AbstractFit(object):
    """General class for fitting of a model with a set of parameters to a dataset with x- and y data"""

    def __init__(self, model, x, y):
        self.model = model
        self.x = x
        self.y = y

        self.val = None

    def fit_parameters(self, parameters, bounds=None, constraint=True, basin_hop=True, T=0.1, **kwargs):
        def objective(par_values, par_names, model, x, y):
            par_dict = {par_name: par_value for par_name, par_value in zip(par_names, par_values)}
            y_model = model(x, **par_dict)

            yn = y / y.max()

            return np.sum(yn*((y - y_model)**2))

        bounds = self.model.get_bounds(parameters, parameters if type(bounds) == bool else bounds) if bounds else None

        method = kwargs['method'] if 'method' in kwargs else 'Powell' if not bounds else None
        verbose = kwargs.pop('verbose', False)
        par_values = np.array([getattr(self.model, par).value for par in parameters.split(' ')])
        constraints = self.model.get_constraints(parameters) if constraint else None

        #todo this needs some checking if not all parameters are bounded
        def _accept_test(bounds, **kwargs):
            par_values = kwargs['x_new']
            bools = [(-np.inf if pmin is None else pmin) <= val <= (np.inf if pmax is None else pmax)
                     for (pmin, pmax), val in zip(bounds, par_values)]

            return np.all(bools)

        accept_test = partial(_accept_test, bounds)

        if basin_hop:
            result = basinhopping(objective, par_values,
                                  minimizer_kwargs={
                                      'args': (parameters.split(' '), self.model, self.x, self.y),
                                      'bounds': bounds,
                                      'method': method,
                                      'constraints': constraints,
                                      'options': {'disp': verbose},
                                      **kwargs},
                                  T=T,
                                  stepsize=1,
                                  niter=200,
                                  accept_test=accept_test
                                  )

        else:
            result = minimize(objective, par_values, args=(parameters.split(' '), self.model, self.x, self.y),
                              bounds=bounds, method=method, constraints=constraints, options={'disp': verbose}, **kwargs)

        try:
            res_dict = {key: val for key, val in zip(parameters.split(' '), result.x)}
        except TypeError:
            res_dict = {key: val for key, val in zip(parameters.split(' '), [result.x])}

        self.val = result.fun
        return res_dict, result.fun

    def execute(self, parameters, bounds=True, constraint=True, basin_hop=True, T=0.0001):
        res, v = self.fit_parameters(parameters, bounds=bounds, constraint=constraint, basin_hop=basin_hop, T=T)
        self.model.sub_par(res)
        res, v = self.fit_parameters(parameters, bounds=bounds, constraint=constraint, basin_hop=basin_hop, T=T)
        return res, v

    def fit_stepwise(self, bounds=None, **kwargs):
        i = 0
        j = 0
        prev_val = 0

        imax = kwargs.get('imax', 3)
        jmax = kwargs.get('jmax', 20)

        assert imax > 0
        assert jmax > 0

        if bounds:
            assert type(bounds) == bool
        while i < imax and j < jmax:
            j += 1
            res, val = self.fit_parameters('r1 r2', bounds=bounds, **kwargs)
            print(res, val)
            res, val = self.fit_parameters('a1 a2', bounds=bounds, **kwargs)
            print(res, val)
            print('Current minimize value: {}'.format(val))
            if prev_val == val:
                i += 1
            prev_val = val

        self.val = val
        return res, val


class Optimizer(object):
    """ Class for cell coordinate optimizing
    """

    defaults = {
        'binary': 'binary',
        'storm': 'leastsq',
        'brightfield': 'sim_cell',
        'fluorescence': 'sim_cell',
    }

    def __init__(self, cell_obj, data_name='binary', objective=None):
        self.cell_obj = cell_obj
        self.data_name = data_name
        self.val = None
        dclass = self.data_elem.dclass
        objective = self.defaults[dclass] if not objective else objective

        if callable(objective):
            self.objective = objective
        elif type(objective) == str:
            self.objective = objectives_dict[dclass][objective]


        #todo check if this is actually nessecary
        if dclass == 'storm':
            self.r = Parameter('r', value=cell_obj.coords.r,
                               min=cell_obj.coords.r/2, max=cell_obj.coords.r*1.5)
        else:
            self.r = Parameter('r', value=cell_obj.coords.r, min=cell_obj.coords.r / 4, max=cell_obj.coords.r * 4)

        self.xl = Parameter('xl', value=cell_obj.coords.xl,
                            min=cell_obj.coords.xl - cfg.ENDCAP_RANGE / 2, max=cell_obj.coords.xl + cfg.ENDCAP_RANGE / 2)
        self.xr = Parameter('xr', value=cell_obj.coords.xr,
                            min=cell_obj.coords.xr - cfg.ENDCAP_RANGE / 2, max=cell_obj.coords.xr + cfg.ENDCAP_RANGE / 2)
        self.a0 = Parameter('a0', value=cell_obj.coords.coeff[0], min=0, max=cell_obj.data.shape[0]*1.5)
        self.a1 = Parameter('a1', value=cell_obj.coords.coeff[1], min=-5, max=5)
        self.a2 = Parameter('a2', value=cell_obj.coords.coeff[2], min=-0.5, max=0.5)

    @property
    def data_elem(self):
        return self.cell_obj.data.data_dict[self.data_name]

    def get_bounds(self, parameters, bounded):
        """Get the bounds for given parameters.

        Args:
            parameters (:obj:`str`): Parameters separated by spaces for which to get the bounds
            bounded (:obj:`str`): Names of parameters which should be bounded. Should be a string where parameter names
                are separated by spaces.

        Returns:

        """
        bounds = [(getattr(self, par).min, getattr(self, par).max) if par in bounded.split(' ') else (None, None)
                  for par in parameters.split(' ')]

        if len(bounds) == 0:
            return None
        elif np.all(np.array(bounds) == (None, None)): #todo for diff evolution this will give problems
            return None
        else:
            return bounds

    def optimize_parameters(self, parameters, solver='normal', bounds=None, obj_kwargs=None, solver_kwargs=None, **kwargs):

        solver_kwargs = {} if solver_kwargs is None else solver_kwargs
        fun = partial(self.objective, **obj_kwargs) if obj_kwargs else self.objective
        bounds = True if solver == 'DE' else bounds
        bounds = self.get_bounds(parameters, parameters if type(bounds) == bool else bounds) if bounds else None
        method = kwargs['method'] if 'method' in kwargs else 'Powell' if not bounds else None #todo maybe differnt default
        verbose = kwargs.pop('verbose', False)
        par_values = np.array([getattr(self.cell_obj.coords, par) for par in parameters.split(' ')])

        if solver == 'DE':
            result = differential_evolution(fun, bounds,
                                            args=(parameters.split(' '), self.cell_obj, self.data_name),
                                            **solver_kwargs)

        elif solver == 'basin_hop':
            def _accept_test(bounds, **kwargs):
                par_values = kwargs['x_new']
                bools = [(-np.inf if pmin is None else pmin) <= val <= (np.inf if pmax is None else pmax)
                         for (pmin, pmax), val in zip(bounds, par_values)]

                return np.all(bools)

            accept_test = partial(_accept_test, bounds)
            stepsize = solver_kwargs.pop('stepsize', 1)
            niter = solver_kwargs.pop('niter', 200)
            result = basinhopping(fun, par_values,
                                  minimizer_kwargs={
                                      'args': (parameters.split(' '), self.cell_obj, self.data_name),
                                      'bounds': bounds,
                                      'method': method,
                                      'options': {'disp': verbose},
                                      **kwargs},
                                  stepsize=stepsize,
                                  niter=niter,
                                  accept_test=accept_test,
                                  **solver_kwargs)

        elif solver == 'normal':
            result = minimize(fun, par_values, args=(parameters.split(' '), self.cell_obj, self.data_name),
                              method=method, bounds=bounds, options={'disp': verbose}, **kwargs)
        else:
            raise ValueError("Value for 'solver' must be either 'DE', 'basin_hop' or 'normal'")

        try:
            res_dict = {key: val for key, val in zip(parameters.split(' '), result.x)}
        except TypeError:
            res_dict = {key: val for key, val in zip(parameters.split(' '), [result.x])}

        self.sub_par(res_dict)
        self.val = result.fun
        return res_dict, result.fun

    def optimize_parameters_bak(self, parameters, bounds=None, obj_kwargs=None, **kwargs):

        fun = partial(self.objective, **obj_kwargs) if obj_kwargs else self.objective
        bounds = self.get_bounds(parameters, parameters if type(bounds) == bool else bounds) if bounds else None
        method = kwargs['method'] if 'method' in kwargs else 'Powell' if not bounds else None #todo maybe differnt default
        verbose = kwargs.pop('verbose', False)
        par_values = np.array([getattr(self.cell_obj.coords, par) for par in parameters.split(' ')])

     #   bounds = [(-100, 100)] * len(parameters.split(' '))
        print(parameters)
        print(bounds)
        result = differential_evolution(fun, bounds,
                                        args=(parameters.split(' '), self.cell_obj, self.data_name)

                              )
        #
        #
        # result = minimize(fun, par_values, args=(parameters.split(' '), self.cell_obj, self.data_name),
        #                   method=method, bounds=bounds, options={'disp': verbose}, **kwargs)

        try:
            res_dict = {key: val for key, val in zip(parameters.split(' '), result.x)}
        except TypeError:
            res_dict = {key: val for key, val in zip(parameters.split(' '), [result.x])}

        self.sub_par(res_dict)
        self.val = result.fun
        return res_dict, result.fun

    def optimize_stepwise(self, bounds=None, **kwargs):
        i = 0
        j = 0
        prev_val = 0

        imax = kwargs.get('imax', 3)
        jmax = kwargs.get('jmax', 20)

        assert imax > 0
        assert jmax > 0

        if bounds:
            assert type(bounds) == bool
        while i < imax and j < jmax:
            #todo checking and testng
            j += 1
            res, val = self.optimize_parameters('r', bounds=bounds, **kwargs)
            res, val = self.optimize_parameters('xl xr', bounds=bounds, **kwargs)
            res, val = self.optimize_parameters('a0 a1 a2', bounds=bounds, **kwargs)
            print('Current minimize value: {}'.format(val))
            if prev_val == val:
                i += 1
            prev_val = val

        self.val = val
        return res, val

    def optimize(self, bounds=None, **kwargs):
        res, val = self.optimize_parameters('r xl xr a0 a1 a2', bounds=bounds, **kwargs)
        self.val = val
        return res, val

    def sub_par(self, par_dict):
        self.cell_obj.coords.sub_par(par_dict)


def minimize_storm_photons(par_values, par_names, cell_obj, data_name):
    par_dict = {par_name: par_value for par_name, par_value in zip(par_names, par_values)}
    cell_obj.coords.sub_par(par_dict)
    storm_data = cell_obj.data.data_dict[data_name]
    r_vals = cell_obj.coords.calc_rc(storm_data['x'], storm_data['y'])
    bools = r_vals < cell_obj.coords.r

    p = np.sum(storm_data['intensity'][bools])

    return -p / cell_obj.area


def minimize_storm_points(par_values, par_names, cell_obj, data_name):
    par_dict = {par_name: par_value for par_name, par_value in zip(par_names, par_values)}
    cell_obj.coords.sub_par(par_dict)
    storm_data = cell_obj.data.data_dict[data_name]
    r_vals = cell_obj.coords.calc_rc(storm_data['x'], storm_data['y'])
    bools = r_vals < cell_obj.coords.r

    p = np.sum(bools.astype(int))

    return -p / cell_obj.area


def minimize_storm_leastsq(par_values, par_names, cell_obj, data_name, r_upper=None, r_lower=lambda x: 2*np.std(x)):
    par_dict = {par_name: par_value for par_name, par_value in zip(par_names, par_values)}
    cell_obj.coords.sub_par(par_dict)
    storm_data = cell_obj.data.data_dict[data_name]
    r_vals = cell_obj.coords.calc_rc(storm_data['x'], storm_data['y'])

    b_upper = r_vals < (r_vals.mean() + r_upper(r_vals)) if r_upper else True
    b_lower = r_vals > (r_vals.mean() - r_lower(r_vals)) if r_lower else True

    b = np.logical_and(b_upper, b_lower)

    r_vals = r_vals[b]

    return np.sum(np.square(r_vals - cell_obj.coords.r))**2


def minimize_binary_xor(par_values, par_names, cell_obj, data_name):
    par_dict = {par_name: par_value for par_name, par_value in zip(par_names, par_values)}
    cell_obj.coords.sub_par(par_dict)
    binary = cell_obj.coords.rc < cell_obj.coords.r

    return np.sum(np.logical_xor(cell_obj.data.data_dict[data_name], binary))


def minimize_sim_cell(par_values, par_names, cell_obj, data_name):
    par_dict = {par_name: par_value for par_name, par_value in zip(par_names, par_values)}
    r = par_dict.pop('r', cell_obj.coords.r)
    r = r / cell_obj.coords.r

    cell_obj.coords.sub_par(par_dict)
    #todo check and make sure that the r_dist isnt calculated to far out which can give some strange results
    simulated = cell_obj.reconstruct_cell(data_name, r_scale=r)
    real = cell_obj.data.data_dict[data_name]

    #print('sim', simulated[:10, 0])

    cost = np.sum((simulated - real) ** 2)
    #print(par_values, cost)

    return np.sum((simulated - real)**2)


objectives_dict = {
    'binary': {
        'binary': minimize_binary_xor
    },
    'storm': {
        'photons': minimize_storm_photons,
        'points': minimize_storm_points,
        'leastsq': minimize_storm_leastsq
    },
    'brightfield': {
        'sim_cell': minimize_sim_cell
    },
    'fluorescence': {
        'sim_cell': minimize_sim_cell
    }
}

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