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
        self.r = Parameter('r', value=cell_obj.coords.r,
                           min=cell_obj.coords.r/2, max=cell_obj.coords.r*1.5)
        self.xl = Parameter('xl', value=cell_obj.coords.xl,
                            min=cell_obj.coords.xl - cfg.ENDCAP_RANGE / 2, max=cell_obj.coords.xl + cfg.ENDCAP_RANGE / 2)
        self.xr = Parameter('xr', value=cell_obj.coords.xr,
                            min=cell_obj.coords)
        self.a0 = Parameter('a0', value=cell_obj.coords.coeff[0])
        self.a1 = Parameter('a1', value=cell_obj.coords.coeff[1])
        self.a2 = Parameter('a2', value=cell_obj.coords.coeff[2])


class STORMOptimizer(OptimizerBase):
    """Optimizes cell coordinates based on STORM data
    
    Args:
        cell_obj: The Cell object's coordinates to optimize based on STORM data
    Kwargs:
        maximize: {'photons', 'localization'} Whether to maximize number of photons or number of localizations
            per area
        
    
    """

    def __init__(self, cell_obj, method='photons', verbose=True):
        super(STORMOptimizer, self).__init__(cell_obj)
        """

        """
        self.cell_obj = cell_obj
        self.method = method

    def optimize_r(self, src='storm'):
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
        min = minimize(minimize_func, r_guess, args=(self.cell_obj, self.method), method='Powell')
        self.cell_obj.coords.r = min.x
        print('r', min.fun)
        return min.x, min.fun

    def optimize_endcaps(self, src='storm'):
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
        min = minimize(minimize_func, x_lr, args=(self.cell_obj, self.method), method='Powell')
        self.cell_obj.coords.xl, self.cell_obj.coords.xr = x_lr
        print('endcaps', min.fun)
        return min.x, min.fun

    def optimize_fit(self, src='storm'):
        def minimize_func(coeff, cell_obj, maximize):
            cell_obj.coords.coeff = coeff
            storm_data = cell_obj.data.data_dict[src]

            r_vals = cell_obj.coords.calc_rc(storm_data['x'], storm_data['y'])
            bools = r_vals < cell_obj.coords.r

            if maximize == 'photons':
                p = np.sum(storm_data['intensity'][bools])
            elif maximize == 'points':
                p = np.sum(bools)

            return -p/cell_obj.area

        coeff = self.cell_obj.coords.coeff
        min = minimize(minimize_func, coeff, args=(self.cell_obj, self.method), method='Powell')
        self.cell_obj.coords.coeff = coeff
        print('fit', min.fun)
        return min.x, min.fun

    def optimize_overall(self, src='storm', verbose=False):
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

            # print(p)
            # print(len(bools))
            return -p/cell_obj.area
        bounds = [(5, 10), (0, 20), (30, 40), (5, 25), (1e-3, None), (1e-10, 10)]
        par = np.array([self.cell_obj.coords.r, self.cell_obj.coords.xl, self.cell_obj.coords.xr] + list(self.cell_obj.coords.coeff))

        min = minimize(minimize_func, par, args=(self.cell_obj, src, self.method), bounds=bounds,
                       options={'disp': verbose}
                        )
        # min = minimize(minimize_func, par, args=(self.cell_obj, src, self.method), method='Powell',
        #                options={'disp': verbose}
        #                )
        self.cell_obj.coords.r, self.cell_obj.coords.xl, self.cell_obj.coords.xr = min.x[:3]
        self.cell_obj.coords.coeff = np.array(min.x[3:])
        print('overall', min.fun)
        return min.x, min.fun

    def optimize_stepwise(self):
        i = 0
        j = 0
        diff_prev = 0
        while i <3 and j < 10:
            j += 1
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
