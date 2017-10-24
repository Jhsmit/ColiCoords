==========
ColiCoords
==========

Because real cells have curves.

Project Goals
=============

ColiCoords is a python project aimed to make analysis of fluorescence data from rodlike cells more streamlined and intuative. These goals are achieved by describing the shape of the cell by a 2nd degree polynomial, and this simple mathematical description together with a data structure to organize cell data on a single cell basis allows for straightforward and detailed analysis. 

Using ColiCoords
----------------

Lets start out with a simple example where we start out with a binary image of a horizontally oriented cell as well as a corresponding fluorescence image. To turn this data into a ``Cell`` object we need to first create a ``Data`` class and use the ``add_data`` method to add the images as a ``np.ndarray``, as well as indicate the data class (binary, brightfield, fluorescence or storm). This data class is used to initialize a ``Cell`` object. 

.. figure:: /examples/example1/bin_flu_combined.png     

Binary (left) and fluorescence (right) input images.
 
.. code-block:: python

  import tifffile
  from colicoords import Data, Cell

  binary_img = tifffile.imread('binary_1.tif')
  fluorescence_img = tifffile.imread('fluorescence_1.tif')

  data = Data()
  data.add_data(binary_img, 'binary')
  data.add_data(fluorescence_img, 'fluorescence', name='flu_514')

  cell_obj = Cell(data)

The ``Cell`` object has two main attributes: ``data`` and ``coords``. The ``data`` attribute is the instance of ``Data`` used to initialize the ``Cell`` object and holds all images as ``np.ndarray`` subclasses in ``data_dict``. The ``coords`` attribute is an instance of ``Coordinates`` and is used to optimize the cell's coordinate system and perform related calculations. The coordinate system described by a polynomial of second degree together with left and right bounds and the radius of the cell. These values are first inital guesses based on the binary image but can be optimized iteratively:


.. code-block:: python

  cell_object.optimize()
  
  
We can now use the cell's coordinate system together with any fluorescence data (image-based or storm) to plot fuorescence distrubtion along any axis, or calculate properties related to the cells shape, such as radius and lenght as well as area and volume. The units used in the inner workings or ``ColiCoords`` are pixels, in colicoords/config.py the pixel size can be defined which is used for output graphs. 


.. code-block:: python
  
  from colicoords.plot import CellPlot
  import matplotlib.pyplot as plt
  
  cp = CellPlot(cell_obj)
  
  plt.figure()
  plt.imshow(cell_obj.data.data_dict['flu_514'], cmap='viridis', interpolation='nearest')
  cp.plot_outline(coords='mpl')
  cp.plot_midline(coords='mpl')
  plt.show()
  
.. figure:: /examples/example1/fluorescence_outline.png
    
  Fluorescence image with cell midline and outline
  
This shows the fluorescence data together with the cell outline and midline optimized from the binary image, which was obtained from the brightfield image. For an explanation on ColiCoords different coordinate systems please see somesectionthatdoesntexistyet. This makes the cell outline appear larger than the cell in the fluorescence image. Since its only the shape and position of the cell that is important for the coordinate system this will not influence the final results. Furthermore, if the fluorescence signal stains the whole cell this can be used as well for coordinate system optimization - see advanced usage for more details. 

To plot the radial distribution of the ``flu_514`` fluorescence channel:


.. code-block:: python
  f, (ax1, ax2) = plt.subplots(1, 2)
  cp.plot_dist(ax=ax1)
  cp.plot_dist(ax=ax2, norm_x=True, norm_y=True)
  plt.tight_layout()
  
.. figure:: /examples/example1/r_dist.png

Radial distribution curve of fluorescence as measured (left) and normalized (right).
  
The displayed curve is basically a histogram of mean intensity of all fluorescence pixels binned by their distance from the cell midline. When using the ``plot_dist`` method on ``CellPlot`` the bin size is chosen automatically as defined in the config. It is also possible to directly access the data from the ``Cell`` object by calling ``r_dist()``. The radial distribution curves can be normalized in both ``x`` and ``y`` directions. When normalized in the ``x`` direction the radius obtained from the brightfield image is set to one, thereby eliminating cell-to-cell variations in width. 

ColiCoords for many Cell objects
--------------------------------

Of course, you will want to analyze not just one but tens of thousands single cells. And they don't come out of the microscope neatly horizontally aligned and on a one cell per image basis. This is what the ``data_to_cells`` method is for. You will need segmented images - labelled binary - in order for this method to work. This you will have to do yourself by either classical methods (thresholding, watershed) or using machine learning software such as Ilastik_ or MicronML_

.. _Ilastik: http://ilastik.org/
.. _MicronML: http://MicronML.org/

.. code-block:: python

  import tifffile
  from colicoords import Cell, Data
  from colicoords.preprocess import data_to_cells
  from colicoords.plot import CellPlot, CellListPlot
  import matplotlib.pyplot as plt

  binary_stack = tifffile.imread('binary_stack_2.tif')
  flu_stack = tifffile.imread('fluorescence_stack_2.tif')
  brightfield_stack = tifffile.imread('brightfield_stack_2.tif')

  data = Data()
  data.add_data(binary_stack, 'binary')
  data.add_data(flu_stack, 'fluorescence')
  data.add_data(brightfield_stack, 'brightfield')
  
The data class can also hold a stack of images provided all image shapes match. The data class can then be iterated over returning an new instance of ``Data`` with a single slice of each data element. The ``Data`` class also supports indexing analogues to ``np.ndarrays``.

.. code-block:: python

  data_slice = data[5:10, 0:100, 0:100]
  print(data.shape)
  print(data_slice.shape)
  >>> (20, 512, 512)
  >>> (20, 100, 100)
  
This particular slicing operation selects images 5 through 10 and takes the upper left 100x100 square. STORM data is automatically sliced accordingly if its present in the data class. This slicing functionality is used by the ``data_to_cells`` method to obtain single-cell objects.

.. code-block:: python
  cell_list = data_to_cells(data)
  cell_list.optimize(verbose=False)
  

The returned object is a ``CellList`` object which is basically a list of ``Cell`` objects. Many of the single-cell properties can be accessed in the form of a list or array for the whole set of cells. ``CellListPlot`` can be used to easily plot fluorescence distribution of the set of cells or histogram certain properties. 

.. code-block:: python
  fig, axes = plt.subplots(2, 2)
  clp.hist_property(ax=axes[0,0], tgt='radius')
  clp.hist_property(ax=axes[0,1], tgt='length')
  clp.hist_property(ax=axes[1,0], tgt='area')
  clp.hist_property(ax=axes[1,1], tgt='volume')
  plt.tight_layout()

