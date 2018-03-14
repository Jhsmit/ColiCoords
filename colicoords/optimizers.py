import numpy as np
from scipy.optimize import minimize, minimize_scalar
from colicoords.config import cfg
from functools import partial

class Parameter(object):
    def __init__(self, name, value=1, min=1.e-10, max=None):
        self.name = name
        self.min = min
        self.max = max
        self.value = value


class OptimizerBase(object):
    """ Base class for cell coordinate optimizers
    """
    def __init__(self, cell_obj, data_name='binary', objective='binary', multiprocess=False):
        self.cell_obj = cell_obj
        self.data_name = data_name

        dclass = self.data_elem.dclass

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

        while i < imax and j < jmax:
            #todo checking and testng
            j += 1
            res, val = self.optimize_r(bounds=bounds, **kwargs)
            res, val = self.optimize_endcaps(bounds=bounds, **kwargs)
            res, val = self.optimize_fit(bounds=bounds, **kwargs)
            print('Current minimize value: {}'.format(val))
            if prev_val == val:
                i += 1
            prev_val = val

        return res, val

    def optimize_r(self, bounds=None, **kwargs):
        res_dict, val = self.optimize_parameters('r', bounds=bounds, **kwargs)
        return res_dict, val

    def optimize_endcaps(self, bounds=None, **kwargs):
        res_dict, val = self.optimize_parameters('xl xr', bounds=bounds, **kwargs)
        return res_dict, val

    #todo refactor fit its not a fit anymore
    def optimize_fit(self, bounds=None, **kwargs):
        res_dict, val = self.optimize_parameters('a0 a1 a2', bounds=bounds, **kwargs)
        return res_dict, val

    def optimize_overall(self, bounds=None, **kwargs):
        res_dict, val = self.optimize_parameters('r xl xr a0 a1 a2', bounds=bounds, **kwargs)
        return res_dict, val

    def sub_par(self, par_dict):
        self.cell_obj.coords.sub_par(par_dict)


class OptimizerBasedep(object):
    """ Base class for cell coordinate optimizers 
    """
    def __init__(self, cell_obj):
        self.cell_obj = cell_obj

        self.r = Parameter('r', value=cell_obj.coords.r,
                           min=cell_obj.coords.r/2, max=cell_obj.coords.r*1.5)
        self.xl = Parameter('xl', value=cell_obj.coords.xl,
                            min=cell_obj.coords.xl - cfg.ENDCAP_RANGE / 2, max=cell_obj.coords.xl + cfg.ENDCAP_RANGE / 2)
        self.xr = Parameter('xr', value=cell_obj.coords.xr,
                            min=cell_obj.coords.xr - cfg.ENDCAP_RANGE / 2, max=cell_obj.coords.xr + cfg.ENDCAP_RANGE / 2)
        self.a0 = Parameter('a0', value=cell_obj.coords.coeff[0], min=0)
        self.a1 = Parameter('a1', value=cell_obj.coords.coeff[1])
        self.a2 = Parameter('a2', value=cell_obj.coords.coeff[2])

    def get_bounds(self, parameters, bounded):
        bounds = [(getattr(self, par).min, getattr(self, par).max) if par in bounded.split(' ') else (None, None)
                  for par in parameters.split(' ')]

        if len(bounds) == 0:
            return None
        elif np.all(np.array(bounds) == (None, None)):
            return None
        else:
            return bounds

    def optimize_stepwise(self, data_name='storm', minimize_func='leastsq', bounds=None, **kwargs):
        i = 0
        j = 0
        prev_val = 0

        imax = kwargs.get('imax', 3)
        jmax = kwargs.get('jmax', 20)

        while i < imax and j < jmax:
            #todo checking and testng
            j += 1
            res, val = self.optimize_r(data_name=data_name, minimize_func=minimize_func, bounds=bounds, **kwargs)
            res, val = self.optimize_endcaps(data_name=data_name, minimize_func=minimize_func, bounds=bounds, **kwargs)
            res, val = self.optimize_fit(data_name=data_name, minimize_func=minimize_func, bounds=bounds, **kwargs)
            print('Current minimize value: {}'.format(val))
            if prev_val == val:
                i += 1
            prev_val = val

        return res, val

    def optimize_r(self, **kwargs):
        raise NotImplementedError()

    def optimize_endcaps(self, **kwargs):
        raise NotImplementedError()

    def optimize_fit(self, **kwargs):
        raise NotImplementedError()

    def sub_par(self, par_dict):
        self.cell_obj.coords.sub_par(par_dict)


class STORMOptimizer(OptimizerBase):
    """Optimizes cell coordinates based on STORM data
    
    Args:
        cell_obj: The Cell object's coordinates to optimize based on STORM data
    Kwargs:
        maximize: {'photons', 'localization'} Whether to maximize number of photons or number of localizations
            per area
        
    
    """

    def __init__(self, cell_obj):
        #todo method here needs to be refactored, maybe it should also be a kwarg on the individual functions?
        super(STORMOptimizer, self).__init__(cell_obj)
        """

        """
        self.func_dict = {
            'photons': minimize_storm_photons,
            'points': minimize_storm_points,
            'leastsq': minimize_storm_leastsq
        }
        self.default_func = minimize_storm_leastsq

        #Default bounds for r are (Arrrh matey) a bit more stringent.
        self.r = Parameter('r', value=cell_obj.coords.r, min=cell_obj.coords.r/2, max=cell_obj.coords.r*1.5)

    def optimize_r(self, data_name='storm', minimize_func='photons', bounds=None, **kwargs):
        res_dict, val = self.optimize_parameters('r',
                                                 minimize_func=minimize_func, data_name=data_name, bounds=bounds,
                                                 **kwargs)

        return res_dict, val

    def optimize_endcaps(self, data_name='storm', minimize_func='photons', bounds=None, **kwargs):
        res_dict, val = self.optimize_parameters('r xl xr a0 a1 a2',
                                                 minimize_func=minimize_func, data_name=data_name, bounds=bounds,
                                                 **kwargs)

        return res_dict, val

    def optimize_fit(self, data_name='storm', minimize_func='photons', bounds=None, **kwargs):
        res_dict, val = self.optimize_parameters('a0 a1 a2',
                                                 minimize_func=minimize_func, data_name=data_name, bounds=bounds,
                                                 **kwargs)

        return res_dict, val

    def optimize_overall_old(self, src='storm', verbose=False):
        def minimize_func(par, cell_obj, src, maximize):
            r, cell_obj.xl, cell_obj.xr = par[:3]
            cell_obj.coords.coeff = par[3:]
            storm_data = cell_obj.data.data_dict[src]
            r_vals = cell_obj.coords.calc_rc(storm_data['x'], storm_data['y'])
            bools = r_vals < r

            if maximize == 'photons':
                p = np.sum(storm_data['intensity'][bools])
            elif maximize == 'points':
                p = np.sum(bools)

            return -p / cell_obj.area

        bounds = [(5, 10), (0, 20), (30, 40), (5, 25), (1e-3, None), (1e-10, 10)]
        par = np.array([self.cell_obj.coords.r, self.cell_obj.coords.xl, self.cell_obj.coords.xr] + list(
            self.cell_obj.coords.coeff))

        min = minimize(minimize_func, par, args=(self.cell_obj, src, self.method), bounds=bounds,
                       options={'disp': verbose}
                       )
        # min = minimize(minimize_func, par, args=(self.cell_obj, src, self.method), method='Powell',
        #                options={'disp': verbose}
        #                )
        self.cell_obj.coords.r, self.cell_obj.coords.xl, self.cell_obj.coords.xr = min.x[:3]
        self.cell_obj.coords.coeff = np.array(min.x[3:])
        print(self.cell_obj.coords.coeff)

        return min.x, min.fun

    #todo pass **kwargs to scipy minimize from cell.optimize
    def optimize_overall(self, data_name='storm', minimize_func='photons', bounds=None, **kwargs):
        res_dict, val = self.optimize_parameters('r xl xr a0 a1 a2',
                                                 minimize_func=minimize_func, data_name=data_name, bounds=bounds,
                                                 **kwargs)

        return res_dict, val

    def optimize_parameters(self, parameters, data_name='storm', minimize_func='photons', bounds=None, **kwargs):
        minimize_func = self.func_dict[minimize_func] if type(minimize_func) == str else minimize_func

        #todo what if bounds = False?
        bounds = self.get_bounds(parameters, parameters if type(bounds) == bool else bounds) if bounds else bounds
        method = kwargs['method'] if 'method' in kwargs else 'Powell' if not bounds else None #todo maybe differnt default
        verbose = kwargs['verbose'] if 'verbose' in kwargs else False

        par_values = np.array([getattr(self.cell_obj.coords, par) for par in parameters.split(' ')])
        result = minimize(minimize_func, par_values, args=(parameters.split(' '), self.cell_obj, data_name),
                          method=method, bounds=bounds, options={'disp': verbose}, **kwargs)

        try:
            res_dict = {key: val for key, val in zip(parameters.split(' '), result.x)}
        except TypeError:
            res_dict = {key: val for key, val in zip(parameters.split(' '), [result.x])}

        self.sub_par(res_dict)
        return res_dict, result.fun


class BinaryOptimizer(OptimizerBase):

    def __init__(self, cell_obj):
        super(BinaryOptimizer, self).__init__(cell_obj)
        self.cell_obj = cell_obj

    def optimize_r(self):
        def minimize_func(r, cell_obj):
            binary = cell_obj.coords.rc < r
            diff = np.sum(np.logical_xor(cell_obj.data.binary_img, binary))
            return diff

        r_guess = self.cell_obj.coords.r
        min = minimize(minimize_func, r_guess, args=self.cell_obj, method='Powell')
        return min.x, min.fun

    def optimize_endcaps(self):
        def minimize_func_xlr(x_lr, cell_obj):
            cell_obj.coords.xl, cell_obj.coords.xr = x_lr
            binary = cell_obj.coords.rc < cell_obj.coords.r
            diff = np.sum(np.logical_xor(cell_obj.data.binary_img, binary))
            return diff

        x_lr = [self.cell_obj.coords.xl, self.cell_obj.coords.xr]  # Initial guesses for endcap coordinates
        min = minimize(minimize_func_xlr, x_lr, args=self.cell_obj, method='Powell')
        return min.x, min.fun

    def optimize_fit(self):
        def minimize_func_fit(coeff, cell_obj):
            cell_obj.coords.coeff = coeff
            binary = cell_obj.coords.rc < cell_obj.coords.r
            diff = np.sum(np.logical_xor(cell_obj.data.binary_img, binary))
            return diff

        coeff = self.cell_obj.coords.coeff
        min = minimize(minimize_func_fit, coeff, args=self.cell_obj, method='Powell', options={'disp': False, 'xtol':1e-1, 'ftol':1e-1})

        return min.x, min.fun

    def optimize_overall(self, method='Powell', verbose=False):
        def minimize_func_overall(par, cell_obj):
              # todo check len
            cell_obj.coords.r, cell_obj.coords.xl, cell_obj.coords.xr = par[:3]
            coeff = np.array(par[3:])
            cell_obj.coords.coeff = coeff

            binary = cell_obj.coords.rc < cell_obj.coords.r
            diff = np.sum(np.logical_xor(cell_obj.data.binary_img, binary))
            return diff

        par = np.array([self.cell_obj.coords.r, self.cell_obj.coords.xl, self.cell_obj.coords.xr] + list(self.cell_obj.coords.coeff))

        min = minimize(minimize_func_overall, par, args=self.cell_obj,
                       method=method, options={'disp': verbose, 'xtol':1e-2, 'ftol':1e-2,})

        return min.x, min.fun


class FluorescenceOptimizer(OptimizerBase):
    pass


class BrightFieldOptimizer(OptimizerBase):
    pass



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


def minimize_storm_leastsq(par_values, par_names, cell_obj, data_name):
    par_dict = {par_name: par_value for par_name, par_value in zip(par_names, par_values)}
    cell_obj.coords.sub_par(par_dict)
    storm_data = cell_obj.data.data_dict[data_name]
    r_vals = cell_obj.coords.calc_rc(storm_data['x'], storm_data['y'])

    b1 = r_vals > (r_vals.mean() - r_vals.std())
    b2 = r_vals < (r_vals.mean() + r_vals.std())

    b = np.logical_and(b1, b2)

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