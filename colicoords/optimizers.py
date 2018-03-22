import numpy as np
from scipy.optimize import minimize
from colicoords.config import cfg
from functools import partial


class Parameter(object):
    def __init__(self, name, value=1, min=1.e-10, max=None):
        self.name = name
        self.min = min
        self.max = max
        self.value = value


class Optimizer(object):
    """ Base class for cell coordinate optimizers
    """

    defaults = {
        'binary': 'binary',
        'storm': 'leastsq'
    }

    def __init__(self, cell_obj, data_name='binary', objective=None):
        self.cell_obj = cell_obj
        self.data_name = data_name

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
        verbose = kwargs['verbose'] if 'verbose' in kwargs else False
        par_values = np.array([getattr(self.cell_obj.coords, par) for par in parameters.split(' ')])

        result = minimize(fun, par_values, args=(parameters.split(' '), self.cell_obj, self.data_name),
                          method=method, bounds=bounds, options={'disp': verbose}, **kwargs)

        try:
            res_dict = {key: val for key, val in zip(parameters.split(' '), result.x)}
        except TypeError:
            res_dict = {key: val for key, val in zip(parameters.split(' '), [result.x])}

        self.sub_par(res_dict)
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

        return res, val

    def optimize(self, bounds=None, **kwargs):
        res, val = self.optimize_parameters('r xl xr a0 a1 a2', bounds=bounds, **kwargs)
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
    },
    'fluorescence': {
    }
}

# might want to wrap the objective functions since they have the parameter substituion part in common
# def _wrap_func(par_values, par_names, cell_obj, data_name, objective, obj_kwargs):
#     par_dict = {par_name: par_value for par_name, par_value in zip(par_names, par_values)}
#     cell_obj.coords.sub_par(par_dict)
#     return objective(cell_obj, data_name, **obj_kwargs)