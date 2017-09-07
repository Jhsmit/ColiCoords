import numpy as np
from scipy.optimize import minimize, minimize_scalar


class OptimizerBase(object):
    """ Base class for cell coordinate optimizers 
    """
    #todo some abstractmethods
    pass


class STORMOptimizer(OptimizerBase):
    """Optimizes cell coordinates based on STORM data
    
    Args:
        cell_obj: The Cell object's coordinates to optimize based on STORM data
    Kwargs:
        maximize: {'photons', 'localization'} Whether to maximize number of photons or number of localizations
            per area
        
    
    """

    def __init__(self, cell_obj, method='photons', verbose=True):
        """
        :param storm_data: structured array with entries x, y, photons. x, y coordinates are in cartesian coords
        :param cell_obj: Cell object
        """
        self.cell_obj = cell_obj
        self.method = method

    def optimize_r(self):
        def minimize_func(r, cell_obj, maximize):
            r_vals = cell_obj.calc_rc(cell_obj.data.storm_data['x'], cell_obj.data.storm_data['y'])
            bools = r_vals < np.abs(r)

            if maximize == 'photons':
                p = np.sum(cell_obj.data.storm_data['photons'][bools])
            elif maximize == 'points':
                p = np.sum(bools)

            cell_obj.coords.r = np.abs(r)
            area = cell_obj.area

            return -p/area

        r_guess = self.cell_obj.coords.r
        min = minimize(minimize_func, r_guess, args=(self.cell_obj, self.method), method='Powell')
        self.cell_obj.coords.r = min.x
        return min.x, min.fun

    def optimize_endcaps(self):
        def minimize_func(x_lr, cell_obj, maximize):
            cell_obj.xl, cell_obj.xr = x_lr
            r_vals = cell_obj.calc_rc(cell_obj.data.storm_data['x'], cell_obj.data.storm_data['y'])
            bools = r_vals < cell_obj.coords.r

            if maximize == 'photons':
                p = np.sum(cell_obj.data.storm_data['photons'][bools])
            elif maximize == 'points':
                p = np.sum(bools)

            return -p/cell_obj.area

        x_lr = [self.cell_obj.coords.xl, self.cell_obj.coords.xr]
        min = minimize(minimize_func, x_lr, args=(self.cell_obj, self.method), method='Powell')
        self.cell_obj.xl, self.cell_obj.xr = x_lr
        return min.x, min.fun

    def optimize_fit(self):
        def minimize_func(coeff, cell_obj, maximize):
            cell_obj.coords.coeff = coeff

            r_vals = cell_obj.get_r(cell_obj.data.storm_data['x'], cell_obj.data.storm_data['y'])
            bools = r_vals < cell_obj.r

            if maximize == 'photons':
                p = np.sum(cell_obj.data.storm_data['photons'][bools])
            elif maximize == 'points':
                p = np.sum(bools)

            return -p/cell_obj.area

        coeff = self.cell_obj.coords.coeff
        min = minimize(minimize_func, coeff, args=(self.cell_obj, self.method), method='Powell')

        return min.x, min.fun

    def optimize_overall(self):
        def minimize_func(par, cell_obj, maximize):
            r, cell_obj.xl, cell_obj.xr = par[:3]
            cell_obj.coords.coeff = par[3:]

            r_vals = cell_obj.calc_rc(cell_obj.data.storm_data['x'], cell_obj.data.storm_data['y'])
            bools = r_vals < r

            if maximize == 'photons':
                p = np.sum(cell_obj.data.storm_data['photons'][bools])
            elif maximize == 'points':
                p = np.sum(bools)

            return -p/cell_obj.area

        par = [self.cell_obj.r, self.cell_obj.xl, self.cell_obj.xr] + list(self.cell_obj.c_coords.coeff)

        min = minimize(minimize_func, par, args=(self.cell_obj, self.method), method='Powell')
        self.cell_obj.coords.r, self.cell_obj.coords.xl, self.cell_obj.coords.xr = min.x[:3]
        self.cell_obj.coords.coeff = np.array(min.x[3:])
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


class BinaryOptimizer(OptimizerBase):

    def __init__(self, cell_obj):
        self.cell_obj = cell_obj

    def optimize_r(self):
        def minimize_func(r, cell_obj):
            print(r)
            binary = cell_obj.coords.rc < r
            diff = np.sum(np.logical_xor(cell_obj.data.binary_img, binary))
            print(diff)
            return diff

        r_guess = self.cell_obj.coords.r
        min = minimize(minimize_func, r_guess, args=self.cell_obj, method='Powell')
        self.cell_obj.coords.r = min.x
        return min.x, min.fun

    def optimize_endcaps(self):
        def minimize_func_xlr(x_lr, cell_obj):
            cell_obj.coords.xl, cell_obj.coords.xr = x_lr
            binary = cell_obj.coords.rc < cell_obj.coords.r
            diff = np.sum(np.logical_xor(cell_obj.data.binary_img, binary))
            return diff

        x_lr = [self.cell_obj.coords.xl, self.cell_obj.coords.xr]  # Initial guesses for endcap coordinates

        min = minimize(minimize_func_xlr, x_lr, args=self.cell_obj, method='Powell')
        self.cell_obj.coords.xl, self.cell_obj.coords.xr = x_lr
        return min.x, min.fun

    def optimize_fit(self):
        def minimize_func_fit(coeff, cell_obj):
            cell_obj.coords.coeff = coeff
            binary = cell_obj.coords.rc < cell_obj.coords.r
            diff = np.sum(np.logical_xor(cell_obj.data.binary_img, binary))
            return diff

        coeff = self.cell_obj.coords.coeff
        min = minimize(minimize_func_fit, coeff, args=self.cell_obj, method='Powell', options={'disp': False, 'xtol':1e-1, 'ftol':1e-1})

        self.cell_obj.coords.coeff = min.x
        return min.x, min.fun

    def optimize_overall(self, method='Powell', verbose=True):
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
        self.cell_obj.coords.r, self.cell_obj.coords.xl, self.cell_obj.coords.xr = min.x[:3]
        self.cell_obj.coords.coeff = np.array(min.x[3:])

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
