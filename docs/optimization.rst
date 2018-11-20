Coordinate Optimization
=======================

Upon initialization of a :class:`~colicoords.cell.Cell` object, the parameters of the coordinate system are initialized
with on initial guesses based on the binary image of the cell. This is only a rough estimation aimed to provide a starting
point for further optimization. In ``ColiCoords``, this fitting and optimization is handled by ``symfit``, a python
package that provides a symbolic and intuitive API for using minimizers from ``scipy`` or other minimization solutions.

The shorthand approach for optimizing the coordinate system is:

.. code-block:: python

  cell.optimize()

By calling :func:`~colicoords.cell.optimize()` the coordinate system is optimized for the current ``Cell`` object by
using the default settings. This means the optimization is performed on the binary image using the ``Powell`` minimizer
algoritm. Although the optimization is fast, different data sources and minimizers can yield more accurate results.

Input data classes and cell functions
-------------------------------------

``ColiCoords`` can optimize the coordinate system of the cell based on all compatible data classes, either binary,
brightfield, fluorescence, or STORM data. All optimization methods are implemented by calculating some property by applying
the current coordinate system on the respective data element, and then comparing this property to the measured data by
calculating the chi-squared.

For example, optimimzation based on the brightfield image can be done as follows:

.. code-block:: python

  cell.optimize('brightfield')

Where it is assumed that the brightfield data element is named `'brightfield'`. The appropriate function that is used for
the optimization is chosen automatically based on the data class and can be supplied optionally by the `cell_function`
keyword argument. Note that optimizaion by brightfield cannot be used to determine the value for the cell's radius parameter,
for this the function :func:`~colicoords.cell.Cell.measure_r` has to be used.

In the case of brightfield optimization, a :class:`~colicoords.data_models.NumericalCellModel` is generated which is used
by ``symfit`` to perform the minimization. When the default `cell_function` (:class:`~colicoords.fitting.CellImageFunction`)
is called it first calculates the radial distribution of the brightfield image, and this radial distribution is then used
to reconstruct the brightfield image. The resulting image is an estimation of the measured brightfield image and by
iterative bootstrapping of this process the optimal parameters can be found. This particular optimization strategy can be
use for any roughly isotropic image - ie a cell image that looks identical in all directions radially outward - and is thus
independent of brightfield image type and focal plane and can also be applied to selected fluorescence images.

The most accurate approach of optimization is by optimizing on a STORM dataset of a membrane marker. Here, the default
`cell_function` used (:class:`~colicorods.fitting.CellSTORMMembraneFunction`) calculates for every localization the distance
`r` to the midline of the cell. This is compared to the current radius parameter of the coordinate system to give the
chi-squared. This fitting is a special case since the dependent data (`y`-data) also depends on the optimization parameter
r. To allow a variable dependent data for fitting, the class :class:`~colicoords.fitting.RadialData` is used, which
mimics a :class:`~numpy.ndarray`, however whose value depends on the current value of the `r` parameter.

Minimizers and bounds
---------------------

Optimization can be done by any ``symfit`` compatible minimizer. All minimizers are imported via ``colicoords.minimizers``.
More details on the different minimizers can be found in the ``symfit`` or ``scipy`` docs.

The default minimizer, ``Powell`` is fast but does not always converge to the global minimum. To increase the probability to
find the global minimum, the minimizer ``DifferentialEvolution`` is used. This minimizer searches the parameter space
defined by bounds on ``Parameter`` objects defined in the model scan for candidate solutions.

The minimizers can be chained together to first do a course optimization followed by global optimization:

.. code-block:: python

  cell.optimize('brightfield', minimizer=[Powell, DifferentialEvolution])

The code above will optimize the coordinate system based on the brightfield image, first by using the ``Powell`` minimizer,
and second by using the ``DifferentialEvolution`` minimizer.

Multiprocessing and high-performance computing
----------------------------------------------

The optimization process can take up to tens of seconds per cell, especially if a global minimizer is used. Although the
process only needs to take place once, the optimization process of several thousands of cells can take too much time to
be conveniently executed on normal desktop PCs. ``ColiCoords`` therefore supports multiprocessing so that the user can
take advantage of parallel high-performance computing. To perform optimization in parallel:

.. code-block:: python
  cells.optimize_mp()

Where `cells` is a :class:`~colicoords.cell.CellList` object. The cells to be divided is equally distributed among the
spawned processes, which is by default equal to the number of physical cores present on the host machine.

Models and advanced usage
------

The default model used is :class:`~colicoords.data_models.NumericalCellModel`. Contrary to typical ``symfit`` workflow,
the :class:`~symfit.Parameter` objects are defined and initialized by the model itself, and then used to make up the
model. To adjust parameter values and bound manually, the user must directly interact with a :class:`~colicoords.fitting.CellFit`
object instead of calling :func:`~colicoords.cell.Cell.optimize`.

.. code-block:: python

  from colicoords import CellFit
  fit = CellFit(cell)
  print(fit.model.params) # [a0, a1, a2, r, xl, xr]
  # Set the minimum bound of the `a0` parameter to 5.
  fit.model.params[0].min = 5
  # Se the value of the `r`parameter to 8.
  fit.model.params[3].value = 8

The fitting can then be executed by calling ``fit.execute()`` as usual.

Custom minimization functions
-------

The minimization function `cell_function` is a subclass of :class:`~colicoords.fitting.CellMinimizeFunctionBase` by default.
This when this object is used it is initialized by ``CellFit`` with the instance of the cell object and the name of the
target data element. These attributes are then accessible in the custom ``__call__`` method of the function object.

The ``__call__`` function must take the coordinate parameters with their values as keyword arguments and should return
the calculated data which is compared to the target data element to calculate the chi-squared. Alternatively, the `target_data`
property can be used, as is done for :class:`~colicorods.fitting.CellSTORMMembraneFunction` to specify a different target.

Alternatively, any custom callable can be given as `cell_function`, as long as it complies with the above requirements.
