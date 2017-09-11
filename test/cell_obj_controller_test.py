from cellcoordinates.gui.controller import CellObjectController
from cellcoordinates.data import Data
from test_functions import generate_testdata
import sys
import tifffile
import numpy as np
import os
from PyQt4 import QtGui
import matplotlib.pyplot as plt

app = QtGui.QApplication(sys.argv)

inst = QtGui.QApplication.instance()
print(inst)

data = generate_testdata()

ctrl = CellObjectController(data, '')
cells = ctrl._create_cell_objects(data, 0.5, 2, 'Binary')

for c in cells:
    plt.imshow(c.data.binary_img)
    plt.show()

ctrl._optimize_coords(cells, dclass='Binary')
print(cells)
