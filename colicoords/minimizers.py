from symfit.core.minimizers import *


#ScipyMinimize.execute = wrap_execute(ScipyMinimize.execute)


# class Powell(ScipyMinimize, BaseMinimizer):
#     """
#     Wrapper around :func:`scipy.optimize.minimize`'s Powell algorithm.
#     """

minimizers = {
    'Powell': Powell,
    'Nelder-Mead': NelderMead,
    'BFGS': BFGS,
    'DE': DifferentialEvolution,
    'SLSQP': SLSQP,
    'COBYLA': COBYLA,
    'L-BFGS-B': LBFGSB,
    'BasinHop': BasinHopping
}