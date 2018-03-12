import numpy as np
from scipy.optimize import minimize, minimize_scalar
from colicoords.config import cfg


class Parameter(object):
    def __init__(self, name, value=1, min=1.e-10, max=None):
        self.name = name
        self.min = min
        self.max = max
        self.value = value


class OptimizerBase(object):
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

    def get_bounds(self, names):
        return [(getattr(self, name).min, getattr(self, name).max) for name in names.split(' ')]

    def optimize_stepwise(self, kwargs):
        i = 0
        j = 0
        diff_prev = 0
        while i <3 and j < 10:
            #todo checking and testng
            j += 1
            v, diff = self.optimize_r(**kwargs)
            self.cell_obj.coords.r = v
            v, diff = self.optimize_endcaps(**kwargs)
            self.cell_obj.coords.xl, self.cell_obj.coords.xr = v
            v, diff = self.optimize_fit(**kwargs)
            self.cell_obj.coords.coeff = v

            print('Current minimize value: {}'.format(diff))
            if diff_prev == diff:
                i += 1
            diff_prev = diff

    def optimize_r(self, kwargs):
        raise NotImplementedError()

    def optimize_endcaps(self, kwargs):
        raise NotImplementedError()

    def optimize_fit(self, kwargs):
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

    def __init__(self, cell_obj, method='photons'):
        #todo method here needs to be refactored, maybe it should also be a kwarg on the individual functions?
        super(STORMOptimizer, self).__init__(cell_obj)
        """

        """
        self.method = method
        #Default bounds for r are (Arrrh matey) a bit more stringent.
        self.r = Parameter('r', value=cell_obj.coords.r, min=cell_obj.coords.r/2, max=cell_obj.coords.r*1.5)

    def optimize_r(self, src='storm', method=None, bounds=None, verbose=False):
        def minimize_func(r, cell_obj, maximize):
            storm_data = cell_obj.data.data_dict[src]
            r_vals = cell_obj.coords.calc_rc(storm_data['x'], storm_data['y'])
            bools = r_vals < np.abs(r)

            if maximize == 'photons':
                p = np.sum(storm_data['intensity'][bools])
            elif maximize == 'points':
                p = np.sum(bools)

            cell_obj.coords.r = np.abs(r)
            area = cell_obj.area

            return -p/area

        r_guess = self.cell_obj.coords.r

        if bounds:
            assert bounds or bounds == 'r'
            bounds = self.get_bounds('r')

        if not method:
            method = 'Powell' if not bounds else None # todo maybe different default method?

        min = minimize(minimize_func, r_guess, args=(self.cell_obj, self.method), method=method, bounds=bounds)
        return min.x, min.fun

    def optimize_endcaps(self, src='storm', method=None, bounds=None, verbose=False):
        def minimize_func(x_lr, cell_obj, maximize):
            cell_obj.coords.xl, cell_obj.coords.xr = x_lr
            storm_data = cell_obj.data.data_dict[src]
            r_vals = cell_obj.coords.calc_rc(storm_data['x'], storm_data['y'])
            bools = r_vals < cell_obj.coords.r

            if maximize == 'photons':
                p = np.sum(storm_data['intensity'][bools])
            elif maximize == 'points':
                p = np.sum(bools)

            return -p/cell_obj.area

        x_lr = [self.cell_obj.coords.xl, self.cell_obj.coords.xr]

        if bounds:
            'xl xr' if type(bounds) == bool else bounds
            bounds = self.get_bounds(bounds)

        if not method:
            method = 'Powell' if not bounds else None  # todo maybe different default method?
        min = minimize(minimize_func, x_lr, args=(self.cell_obj, self.method), method=method, bounds=bounds)
        return min.x, min.fun

    def optimize_fit(self, src='storm', method=None, bounds=None, verbose=False):
        def minimize_func(coeff, cell_obj, maximize):
            cell_obj.coords.coeff = coeff
            storm_data = cell_obj.data.data_dict[src]
            r_vals = cell_obj.coords.calc_rc(storm_data['x'], storm_data['y'])
            bools = r_vals < cell_obj.coords.r

            if maximize == 'photons':
                p = np.sum(storm_data['intensity'][bools])
            elif maximize == 'points':
                p = np.sum(bools)
            else:
                raise ValueError('Invalid maximize parameter')

            return -p/cell_obj.area

        coeff = self.cell_obj.coords.coeff

        if bounds:
            bounds = self.get_bounds('a0 a1 a2' if type(bounds) == bool else bounds)

        if not method:
            method = 'Powell' if not bounds else None  # todo maybe different default method?

        min = minimize(minimize_func, coeff, args=(self.cell_obj, self.method), method=method, bounds=bounds)
        return min.x, min.fun



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
    def optimize_overall(self, src='storm', method=None, bounds=None, verbose=False):
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

            return -p/cell_obj.area

        if bounds:
            bounds = self.get_bounds('r xl xr a0 a1 a2' if type(bounds) == bool else bounds)

        if not method:
            method = 'Powell' if not bounds else None  # todo maybe different default method?

        par = np.array([self.cell_obj.coords.r, self.cell_obj.coords.xl, self.cell_obj.coords.xr] + list(self.cell_obj.coords.coeff))
        result = minimize(minimize_func, par, args=(self.cell_obj, src, self.method), method=method, bounds=bounds,
                          options={'disp': verbose})
        res_dict = {key: val for key, val in zip(['r', 'xl', 'xr', 'a0', 'a1', 'a2'], result.x)}
        return res_dict, result.fun

    def optimize_parameters(self, parameters=None, src='storm', method=None, bounds=None, verbose=False):
        def minimize_func(par_values, par_names, cell_obj, src, maximize):
            par_dict = {par_name: par_value for par_name, par_value in zip(par_names, par_values)}
            cell_obj.coords.sub_par(par_dict)
            storm_data = cell_obj.data.data_dict[src]
            r_vals = cell_obj.coords.calc_rc(storm_data['x'], storm_data['y'])
            bools = r_vals < cell_obj.coords.r

            if maximize == 'photons':
                p = np.sum(storm_data['intensity'][bools])
            elif maximize == 'points':
                p = np.sum(bools)

            return -p/cell_obj.area

        parameters = 'r xl xr a0 a1 a2' if not parameters else parameters
        if bounds:
            bounds = self.get_bounds(parameters if type(bounds) == bool else bounds)

        if not method:
            method = 'Powell' if not bounds else None  # todo maybe different default method?

        par_values = np.array([getattr(self.cell_obj.coords, par) for par in parameters.split(' ')])
        result = minimize(minimize_func, par_values, args=(parameters.split(' '), self.cell_obj, src, self.method), method=method, bounds=bounds,
                          options={'disp': verbose})

        try:
            len(result.x)
            out_values = result.x
        except TypeError:
            out_values = [result.x]

        res_dict = {key: val for key, val in zip(parameters.split(' '), out_values)}
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

    def optimize_stepwise(self):
        i = 0
        diff_prev = 0
        while i <3:
            v, diff = self.optimize_r()
            v, diff = self.optimize_endcaps()
            v, diff = self.optimize_fit()

            print('Current minimize value: {}'.format(diff))
            if diff_prev == diff:
                i += 1
            diff_prev = diff


class FluorescenceOptimizer(OptimizerBase):
    pass

class BrightFieldOptimizer(OptimizerBase):
    pass
