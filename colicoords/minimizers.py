from symfit.core.minimizers import *

def wrap_execute(execute):
    #print('dit is een print')
    def wrapped_function(*args, **kwargs):
        res = execute(*args, **kwargs)
        #print(res)
        #print(res.objective_value)
        return res
    return wrapped_function


ScipyMinimize.execute = wrap_execute(ScipyMinimize.execute)


class Powell(ScipyMinimize, BaseMinimizer):
    """
    Wrapper around :func:`scipy.optimize.minimize`'s Powell algorithm.
    """

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