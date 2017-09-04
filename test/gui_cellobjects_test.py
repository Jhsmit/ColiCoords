from PyQt4 import QtGui
import sys
from cellcoordinates.gui.controller import CellObjectController


app = QtGui.QApplication(sys.argv)



ctrl = CellObjectController()


sys.exit(app.exec_())