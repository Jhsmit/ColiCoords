|travis| |appveyor| |docs| |license| |doi|

|test|

.. |test| image:: images/ColiCoords_Final_Logo.svg
    :width: 50%

.. |travis| image:: https://travis-ci.org/Jhsmit/ColiCoords.svg?branch=master
    :target: https://travis-ci.org/Jhsmit/ColiCoords 
.. |appveyor| image:: https://ci.appveyor.com/api/projects/status/801teey9fnm8kuc9?svg=true
    :target: https://ci.appveyor.com/project/Jhsmit/colicoords
.. |docs| image:: https://readthedocs.org/projects/colicoords/badge/?version=latest
    :target: https://colicoords.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status
.. |license| image:: https://img.shields.io/badge/License-MIT-yellow.svg
    :target: https://opensource.org/licenses/MIT
.. |doi| image:: https://zenodo.org/badge/92830488.svg
   :target: https://zenodo.org/badge/latestdoi/92830488
  
  

Project Goals
=============

ColiCoords is a python project for analysis of fluorescence microscopy data from rodlike cells. The project is aimed to be an open, well documented platform where users can easily share data through compact hdf5 files and analysis pipelines in the form of Jupyter notebooks.


Installation
============

`ColiCoords` is in its initial realease phase and will in time be distributed via conda/pip. For the moment, to install `ColiCoords`, a python 3.6 installation is required as well as dependencies listed in `requirements.txt`. Then download the latest release, and then to install:

.. code:: bash

    python setup.py install


Although `ColiCoords` features automated testing, there are likely to be bugs. Users are encouraged to report them via the Issues page on GitHub. 

Contact: jhsmit@gmail.com

Documentation and Examples
==========================

Preliminary docs: https://colicoords.readthedocs.io/en/latest/index.html

Several examples of `ColiCoords` usage can be found in the examples directory. More examples will be added in time. 


|pipeline|

.. |pipeline| image:: images/pipeline_figure.png
    :width: 100%
