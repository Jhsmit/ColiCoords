import numpy as np
from scipy.optimize import minimize, basinhopping, brute
from colicoords.config import cfg
from functools import partial



class Parameter(object):
    def __init__(self, name, value=1., min=1.e-10, max=None, fixed=False):
        self.name = name
        self.min = min
        self.max = max
        self.value = value
        self.fixed = fixed


class CellFitting(object):
    """Fits cell radial distribution curve to a model"""

    def __init__(self, model, x, y):
        self.model = model
        self.x = x
        self.y = y

    def fit_parameters(self, parameters, bounds=None, constraint=True, basin_hop=True, T=0.1, **kwargs):
        def objective(par_values, par_names, model, x, y):
            par_dict = {par_name: par_value for par_name, par_value in zip(par_names, par_values)}
            y_model = model(x, **par_dict)

            yn = y / y.max()

            return np.sum(yn*((y - y_model)**2))

        bounds = self.model.get_bounds(parameters, parameters if type(bounds) == bool else bounds) if bounds else None

        method = kwargs['method'] if 'method' in kwargs else 'Powell' if not bounds else None
        verbose = kwargs.pop('verbose', False)
       # method = 'Nelder-Mead'
        par_values = np.array([getattr(self.model, par).value for par in parameters.split(' ')])
        constraints = self.model.get_constraints(parameters) if constraint else None

        #todo this needs some checking if not all parameters are bounded
        def _accept_test(bounds, **kwargs):
            par_values = kwargs['x_new']
            bools = [(-np.inf if pmin is None else pmin) <= val <= (np.inf if pmax is None else pmax)
                     for (pmin, pmax), val in zip(bounds, par_values)]

            return np.all(bools)

        accept_test = partial(_accept_test, bounds)

        # ranges = (slice(0, 1.1, 0.1), slice(0, 1.1, 0.1), slice(4, 6.5, 0.05), slice(4, 6.5, 0.05))
        # resbrute = brute(
        #     objective,
        #     ranges,
        #     args=(parameters.split(' '), self.model, self.x, self.y),
        #     full_output=True
        #
        # )


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
      #  return resbrute

        return res_dict, result.fun

    def fit_parameters_old(self, parameters, bounds=None, constraint=True, **kwargs):
        def tempfunc(par_values, par_names, model, x, y):
            par_dict = {par_name: par_value for par_name, par_value in zip(par_names, par_values)}
            y_model = model(x, **par_dict)

            return np.sum((y - y_model)**2)

        bounds = self.model.get_bounds(parameters, parameters if type(bounds) == bool else bounds) if bounds else None
        method = kwargs['method'] if 'method' in kwargs else 'Powell' if not bounds else None
        verbose = kwargs.pop('verbose', False)
        par_values = np.array([getattr(self.model, par).value for par in parameters.split(' ')])
        constraints = self.model.get_constraints(parameters) if constraint else None

        result = minimize(tempfunc, par_values, args=(parameters.split(' '), self.model, self.x, self.y),
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

    def execute1(self, bounds=True, constraint=True):
        # This function assumes the model has parameters a1 a2 r1 r2

        self.model.a1.value = 1.
        self.model.a2.value = 0.01
        res1, val1 = self.fit_parameters('a1 a2 r1 r2', bounds=bounds, constraint=constraint)

        self.model.a1.value = 0.01
        self.model.a2.value = 1.
        res2, val2 = self.fit_parameters('a1 a2 r1 r2', bounds=bounds, constraint=constraint)

        if val1 < val2:
            a1, a2 = res1['a1'], res1['a2']
        else:
            a1, a2 = res2['a1'], res2['a2']

        self.model.a1.value = 0.1#a1
        self.model.a2.value = 1.1#a2

        r_vals = [1, 2, 3, 4, 5, 6, 7, 8]
        rdicts = []
        f_vals = []
        for r in r_vals:
            self.model.r1.value = r
            self.model.r2.value = r + 0.5
            print(r)

            res, val = self.fit_parameters('a1 a2 r1 r2', bounds=bounds, constraint=constraint)
            rdicts.append(res)
            f_vals.append(val)

        print('results')

        for rd, fv, rs in zip(rdicts, f_vals, r_vals):
            print('----')
            print(rd)
            print(fv)
            print(rs)

        r_vals = [3.5, 3.75, 4.0, 4.25, 4.5, 4.75, 5, 5.25, 5.5, 5.75]
        rdicts = []
        f_vals = []
        for r in r_vals:
            self.model.r1.value = r
            self.model.r2.value = r + 0.1
            print(r)

            res, val = self.fit_parameters('a1 a2 r1 r2', bounds=bounds, constraint=constraint)
            rdicts.append(res)
            f_vals.append(val)

        print('results')

        for rd, fv, rs in zip(rdicts, f_vals, r_vals):
            print('----')
            print(rd)
            print(fv)
            print(rs)

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
            #todo checking and testng
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
        self.val = np.inf
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
        self.a0 = Parameter('a0', value=cell_obj.coords.coeff[0], min=0)
        self.a1 = Parameter('a1', value=cell_obj.coords.coeff[1])
        self.a2 = Parameter('a2', value=cell_obj.coords.coeff[2])

    @property
    def data_elem(self):
        return self.cell_obj.data.data_dict[self.data_name]

    def get_bounds(self, parameters, bounded):
        bounds = [(getattr(self, par).min, getattr(self, par).max) if par in bounded.split(' ') else (None, None)
                  for par in parameters.split(' ')]

        if len(bounds) == 0:
            return None
        elif np.all(np.array(bounds) == (None, None)):
            return None
        else:
            return bounds

    def optimize_parameters(self, parameters, bounds=None, obj_kwargs=None, **kwargs):

        fun = partial(self.objective, **obj_kwargs) if obj_kwargs else self.objective
        bounds = self.get_bounds(parameters, parameters if type(bounds) == bool else bounds) if bounds else None
        method = kwargs['method'] if 'method' in kwargs else 'Powell' if not bounds else None #todo maybe differnt default
        verbose = kwargs.pop('verbose', False)
        par_values = np.array([getattr(self.cell_obj.coords, par) for par in parameters.split(' ')])

        result = minimize(fun, par_values, args=(parameters.split(' '), self.cell_obj, self.data_name),
                          method=method, bounds=bounds, options={'disp': verbose}, **kwargs)

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
    simulated = cell_obj.sim_cell(data_name, r_scale=r)
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