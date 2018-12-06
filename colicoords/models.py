import numpy as np
from scipy.integrate import quad
from colicoords.config import cfg
from symfit.core.fit import CallableNumericalModel
from symfit import Parameter, Variable


class NumericalCellModel(CallableNumericalModel):
    """
    ``Symfit`` model to describe the cell used in coordinate optimization.

    Parameters
    ----------
    cell_obj : :class:`~colicoords.cell.Cell`
        Cell object to be modelled.
    cell_function : :obj:`callable`
        Function used to calculate the dependent variable. Usually a subclass of
        :class:~`colicoords.fitting.CellMinimizeFunctionBase` but it can be any callable as long as it accepts the
        coordinate system's parameters as keyword arguments in its `__call__`.
    """
    def __init__(self, cell_obj, cell_function):
        self.cell_obj = cell_obj
        self.cell_function = cell_function

        r = Parameter('r', value=cell_obj.coords.r, min=cell_obj.coords.r / 4, max=cell_obj.coords.r * 4)
        xl = Parameter('xl', value=cell_obj.coords.xl,
                       min=cell_obj.coords.xl - cfg.ENDCAP_RANGE / 2, max=cell_obj.coords.xl + cfg.ENDCAP_RANGE / 2)
        xr = Parameter('xr', value=cell_obj.coords.xr,
                       min=cell_obj.coords.xr - cfg.ENDCAP_RANGE / 2, max=cell_obj.coords.xr + cfg.ENDCAP_RANGE / 2)
        a0 = Parameter('a0', value=cell_obj.coords.coeff[0], min=0, max=cell_obj.data.shape[0]*1.5)
        a1 = Parameter('a1', value=cell_obj.coords.coeff[1], min=-15, max=15)
        a2 = Parameter('a2', value=cell_obj.coords.coeff[2], min=-0.05, max=0.05)

        y = Variable('y')

        parameters = [a0, a1, a2, r, xl, xr]
        super(NumericalCellModel, self).__init__({y: cell_function}, [], parameters)

    def __reduce__(self):
        #todo check if this is still needed.
        return (
            self.__class__,
            (self.cell_obj, self.cell_function)
        )


try:
    from joblib import Memory as JobMemory
    class Memory(JobMemory):
        def __init__(self, *args, **kwargs):
            args = (cfg.CACHE_DIR,) + args
            super(Memory, self).__init__(*args, **kwargs)

except ImportError:
    pass


class PSF(object):
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, x):
        return (1/(self.sigma*np.sqrt(2*np.pi))) * np.exp(-(x/self.sigma)**2 / 2)

#todo psf is an object and depending on which instance it is joblib will recalculate even if the sigma is the same
#todo https://docs.scipy.org/doc/scipy/reference/generated/scipy.LowLevelCallable.html#scipy.LowLevelCallable
def _y1(x, r1, psf, psf_uid):
    def integrant(x, v, r1, psf):
        return psf(x - v) * np.nan_to_num(np.sqrt(r1 ** 2 - x ** 2))

    yarr, yerr = np.array([quad(integrant, -np.inf, np.inf, args=(v, r1, psf)) for v in x]).T
    return yarr


def _y2(x, r2, psf, psf_uid):
    def integrant(x, v, r2, psf):
        try:
            return psf(x - v) * np.nan_to_num(np.sqrt(1 + (x ** 2 / (r2 ** 2 - x ** 2))))
        except ZeroDivisionError:
            return 0

    yarr, yerr = np.array([quad(integrant, -np.inf, np.inf, args=(v, r2, psf)) for v in x]).T
    return yarr


class RDistModel(CallableNumericalModel):
    """
    Symfit model used for modelling of radial distributions.

    Parameters
    ----------
    psf : :obj:`callable`
        Callable describing the point-spread function in 1D.
    mem : :class:`Memory`, optional
        Optional ``joblib`` ``Memory`` object to cache the model's function calls and speed up the fitting process.
    r : :obj:`str`
        Either 'separate' or 'equal'. If 'separate' the model has two radial parameters, `r1`, `r2', corresponding to
        radii of both components. If 'equal' the model has one radial parameter, 'r', for both components.
    """
    def __init__(self, psf, mem=None, r='separate'):
        self.a1 = Parameter(name='a1', value=0.5, min=0)
        self.a2 = Parameter(name='a2', value=0.5, min=0)

        if r == 'separate':
            self.r1 = Parameter(name='r1', value=4.5, min=2, max=6)
            self.r2 = Parameter(name='r2', value=5.5, min=2, max=8)
            parameters = [self.a1, self.a2, self.r1, self.r2]
        elif r == 'equal':
            self.r = Parameter(name='r', value=5.5, min=2, max=8)
            parameters = [self.a1, self.a2, self.r]
        else:
            raise ValueError('Invalid value for r')

        self.x = Variable('x')
        self.y = Variable('y')

        func = RDistFunc(psf, mem)
        self.linear_params = [self.a1, self.a2]
        super(RDistModel, self).__init__({self.y: func}, [self.x], parameters)


class RDistFunc(object):
    """
    Callable that calculates a superposition of membrane and cytsol components returning a radial distribution.

    Parameters
    ----------
    psf : :obj:`callable`
        Callable describing the point-spread function in 1D.
    mem : :class:`Memory`, optional
        Optional ``joblib`` ``Memory`` object to cache the model's function calls and speed up the fitting process.
    """

    def __init__(self, psf, mem=None):
        self.psf = psf

        if mem is not None:
            self.y1 = mem.cache(_y1, ignore=['psf'])
            self.y2 = mem.cache(_y2, ignore=['psf'])
        else:
            self.y1 = _y1
            self.y2 = _y2

        self.i = 10

    def __call__(self, x, **kwargs):
        r = kwargs.pop('r', None)
        r1 = kwargs.pop('r1', r)
        r2 = kwargs.pop('r2', r)
        a1 = kwargs['a1']
        a2 = kwargs['a2']

        if self.i:
            r1_l = int(np.floor(self.i * r1)) / self.i
            r1_u = int(np.ceil(self.i * r1)) / self.i

            if r1_l == r1 and r1_u == r1:
                y1 = self.y1(x, r1, self.psf, self.psf.sigma)
            else:
                y1_l = self.y1(x, r1_l, self.psf, self.psf.sigma)
                y1_u = self.y1(x, r1_u, self.psf, self.psf.sigma)

                y1 = (y1_l*(r1_u - r1) + y1_u*(r1 - r1_l)) / (r1_u - r1_l)

            r2_l = int(np.floor(self.i * r2)) / self.i
            r2_u = int(np.ceil(self.i * r2)) / self.i

            if r2_l == r2 and r2_u == r2:
                y2 = self.y2(x, r2, self.psf, self.psf.sigma)
            else:
                y2_l = self.y2(x, r2_l, self.psf, self.psf.sigma)
                y2_u = self.y2(x, r2_u, self.psf, self.psf.sigma)

                y2 = (y2_l * (r2_u - r2) + y2_u * (r2 - r2_l)) / (r2_u - r2_l)

        else:
            y1 = self.y1(x, r1, self.psf, self.psf.sigma)
            y2 = self.y2(x, r2, self.psf, self.psf.sigma)

        try:
            yarr = (a1 / (0.5 * np.pi * r1 ** 2))*y1 + (a2 / (np.pi * r2))*y2
        except ValueError: # a's are arrays
            assert a1.shape == a2.shape
            yarr = (a1[:, np.newaxis] / (0.5 * np.pi * r1 ** 2))*y1[np.newaxis, :] + (a2[:, np.newaxis] / (np.pi * r2))*y2[np.newaxis, :]

        return yarr
