Configuration
=============

The ``config`` module can be used to alter and ``ColiCoords``' default configuration values. These are mostly default
values in relation to the generation of graphs via the ``plot`` module and they do not affect coordinate transformations.
These default values for plot generation can be overruled by giving explicit keyword arguments to plot functions.

+-------------------+-----------------+-----------------+----------------+---------------------------------------------------------------------------------------------------------------------------+
| Name              | Type            | Default value   | Units          |Description                                                                                                                |
+=====================================+=================+================+===========================================================================================================================+
| IMG_PIXELSIZE     | :obj:`float`    | 80              | Nanometers     | Pixel size of the acquired images.                                                                                        |
+-------------------+-----------------+-----------------+----------------+---------------------------------------------------------------------------------------------------------------------------+
| ENDCAP_RANGE      | :obj:`float`    | 20.0            | Pixels         | Default bounds for positions of cell's poles used in bounded optimization.                                                |
+-------------------+-----------------+-----------------+----------------+---------------------------------------------------------------------------------------------------------------------------+
| R_DIST_STOP       | :obj:`float`    | 20.0            | Pixels         | Upper limit for generation of radial distribution curves.                                                                 |
+-------------------+-----------------+-----------------+----------------+---------------------------------------------------------------------------------------------------------------------------+
| R_DIST_STEP       | :obj:`float`    | 0.5             | Pixels         | Step size between datapoints for generation of radial distribution curves                                                 |
+-------------------+-----------------+-----------------+----------------+---------------------------------------------------------------------------------------------------------------------------+
| R_DIST_SIGMA      | :obj:`float`    | 0.3             | Pixels         | Size of the sigma parameter of gaussian used for convolution to generate radial distribution curves.                      |
+-------------------+-----------------+-----------------+----------------+---------------------------------------------------------------------------------------------------------------------------+
| L_DIST_NBINS      | :obj:`int`      | 100             | -              | Number of bins to generate the longitudinal distribution curves.                                                          |
+-------------------+-----------------+-----------------+----------------+---------------------------------------------------------------------------------------------------------------------------+
| L_DIST_SIGMA      | :obj:`float`    | 0.5             | Pixels         | Size of the sigma parameter of gaussian used for convolution to generate longitudinal distribution curves.                |
+-------------------+-----------------+-----------------+----------------+---------------------------------------------------------------------------------------------------------------------------+
| PHI_DIST_STEP     | :obj:`float`    | 1.0             | Degrees        | Step size between datapoints for generation of angular distribution curves.                                               |
+-------------------+-----------------+-----------------+----------------+---------------------------------------------------------------------------------------------------------------------------+
| PHI_DIST_SIGMA    | :obj:`float`    | 0.5             | Degrees        | Size of the sigma parameter of gaussian used for convolution to generate longitudinal distribution curves.                |
+-------------------+-----------------+-----------------+----------------+---------------------------------------------------------------------------------------------------------------------------+
| CACHE_DIR         | :obj:`str`      |                 |                | Path to the chache dir directory.
+-------------------+-----------------+-----------------+----------------+---------------------------------------------------------------------------------------------------------------------------+
| DEBUG             | :obj:`bool`     | False           |                | Set to ``True`` to print ``numpy`` division warnings.                                                                     |
+-------------------+-----------------+-----------------+----------------+---------------------------------------------------------------------------------------------------------------------------+
