__version__ = "0.1.0"
print('You are using ColiCoords version {}'.format(__version__))

from colicoords.data_models import Data
from colicoords.cell import Cell, CellList
from colicoords.preprocess import data_to_cells
from colicoords.plot import CellPlot, CellListPlot
from colicoords.fileIO import load, save, load_thunderstorm
from colicoords.models import RDistModel, Memory
from colicoords.optimizers import LinearModelFit, AbstractFit, Optimizer
from colicoords.synthetic_data import SynthCell, SynthCellList


try:
    from colicoords.models import Memory
except ImportError:
    pass
