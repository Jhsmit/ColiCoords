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



    def __init__(self, cell_obj, maximize='photons'):
        """

        :param storm_data: structured array with entries x, y, photons. x, y coordinates are in cartesian coords
        :param cell_obj: Cell object
        """
        self.cell_obj = cell_obj


    def optimize_r(self):
        def minimize_func(r, storm_data, cell_obj):
            r_vals = cell_obj.get_r(storm_data['x'], storm_data['y'])
            bools = r_vals < np.abs(r)

            photons = np.sum(storm_data['photons'][bools])
            points = np.sum(bools)
            cell_obj.r = np.abs(r)
            area = cell_obj.area
            # r_g.append(r)
            # val.append(area/photons)
            return -photons/area

        r_guess = 20#1*cell_obj.r # todo check optimizaton of this r
        min = minimize(minimize_func, r_guess, args=(self.storm_data, self.cell_obj), method='Powell', options={'disp':True})
        #min = minimize_scalar(minimize_func, args=(storm_data, cell_obj), options={'disp':True}, method='Bounded', bounds=[30, 500])

        return min.x, min.fun

    def optimize_endcaps(self):
        def minimize_func(x_lr, storm_data, cell_obj):
            cell_obj.xl, cell_obj.xr = x_lr
            r_vals = cell_obj.get_r(storm_data['x'], storm_data['y'])
            bools = r_vals < cell_obj.r

            photons = np.sum(storm_data['photons'][bools])
            area = cell_obj.area
            return -photons / area


        x_lr = [self.cell_obj.xl, self.cell_obj.xr]
        print(x_lr)
        min = minimize(minimize_func, x_lr, args=(self.storm_data, self.cell_obj), method='Powell', options={'disp':True})

        return min.x, min.fun

    def optimize_fit(self):
        def minimize_func(par, storm_data, cell_obj):
            cell_obj.c_coords.params = par

            r_vals = cell_obj.get_r(storm_data['x'], storm_data['y'])
            bools = r_vals < cell_obj.r

            photons = np.sum(storm_data['photons'][bools])
            area = cell_obj.area
            return -photons / area


        par = self.cell_obj.c_coords.params*0.9
        min = minimize(minimize_func, par, args=(self.storm_data, self.cell_obj), method='Powell', options={'disp': True})

        return min.x, min.fun

    def optimize_all(self):
        def minimize_func(arr, storm_data, cell_obj):
            r, xl, xr = arr[:3]
            par = arr[3:]
            cell_obj.xl = xl
            cell_obj.xr = xr
            cell_obj.r = r
            cell_obj.c_coords.params = par

            r_vals = cell_obj.get_r(storm_data['x'], storm_data['y'])
            bools = r_vals < r

            photons = np.sum(storm_data['photons'][bools])
            area = cell_obj.area

            return -photons/area

        arr = [self.cell_obj.r, self.cell_obj.xl, self.cell_obj.xr] + list(self.cell_obj.c_coords.params)
        arr = np.array(arr)
        arr *= 1.
        arr = list(arr)

        min = minimize(minimize_func, arr, args=(self.storm_data, self.cell_obj), method='Powell', options={'disp': True})

        return min.x, min.fun


class BinaryOptimizer(OptimizerBase):
    pass

