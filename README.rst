|test_action| |docs| |binder| |codecov| |license| |doi| 

|image|

.. |image| image:: images/logo_with_cell_1280x640.png
    :width: 50%



.. |test_action| image:: https://github.com/Jhsmit/ColiCoords/workflows/pytest/badge.svg
    :target: https://github.com/Jhsmit/ColiCoords/actions?query=workflow%3Apytest
.. |docs| image:: https://readthedocs.org/projects/colicoords/badge/?version=latest
    :target: https://colicoords.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status
.. |binder| image:: https://mybinder.org/badge_logo.svg 
    :target: https://mybinder.org/v2/gh/Jhsmit/ColiCoords/master
.. |codecov| image:: https://codecov.io/gh/Jhsmit/ColiCoords/branch/master/graph/badge.svg
  :target: https://codecov.io/gh/Jhsmit/ColiCoords
.. |license| image:: https://img.shields.io/badge/License-MIT-yellow.svg
    :target: https://opensource.org/licenses/MIT
.. |doi| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.1412663.svg
   :target: https://doi.org/10.5281/zenodo.1412663

`Documentation <https://colicoords.readthedocs.io/>`_

`PLOS One Paper <https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0217524>`_ 

`BioRxiv Paper <https://www.biorxiv.org/content/10.1101/608109v1>`_

`Paper code <https://github.com/Jhsmit/ColiCoords-Paper>`_

Project Goals
=============

ColiCoords is a python project for analysis of fluorescence microscopy data from rodlike cells. The project is aimed to be an open, well documented platform where users can easily share data through compact hdf5 files and analysis pipelines in the form of Jupyter notebooks.


Installation
============

``ColiCoords`` is available on PyPi and Conda Forge. Currently, python >= 3.6 is required.

Installation by `Conda <https://conda.io/docs/>`_.:

.. code:: bash
     
     conda install -c conda-forge colicoords 

For installation via PyPi a C++ compiler is required for installing the dependency `mahotas  <https://mahotas.readthedocs.io/en/latest/index.html>`_. Alternatively, ``mahotas`` can be installed separately from Conda. 

To install ``ColiCoords`` from pypi:

.. code:: bash

    pip install colicoords


Although `ColiCoords` features automated testing, there are likely to be bugs. Users are encouraged to report them via the Issues page on GitHub. 

Contact: jhsmit@gmail.com

Examples
========

Several examples of `ColiCoords` usage can be found in the examples directory.


|pipeline|

.. |pipeline| image:: images/pipeline_figure.png
    :width: 100%

Citation
========

If you you use ``ColiCoords`` (or any modified version) for scientific publication or other purposes, please cite:

Smit, J. H., Li, Y., Warszawik, E. M., Herrmann, A. & Cordes, T. *ColiCoords: A Python package for the analysis of bacterial fluorescence microscopy data.* PLOS ONE 14, e0217524 (2019).

If you use the ``CNN`` module please also cite:

Falk, T. et al. *U-Net: deep learning for cell counting, detection, and morphometry.* Nat Methods 16, 67–70 (2019).

Acknowledgement
===============

ColiCoords up to v0.1.4 was developed by Jochem Smit within ongoing projects of the `Cordes Lab <http://www.mikrobiologie.biologie.uni-muenchen.de/forschung/ag_cordes/index.html>`_.
The project was financed until 01-08-2018 by an ERC Starting Grant (No. 638536 - SM-IMPORT to Thorben Cordes) and an ERC Advanced Grant (No. 694610 - SUPRABIOTICS to Andreas Herrmann).
