from colicoords.gui.images_select import NavigationWindow, ImageWindow, MPLWindow
from PyQt5.QtCore import pyqtSlot
from PyQt5 import QtCore

from PyQt5 import QtGui
import mahotas as mh
import numpy as np
import os
import math
import seaborn as sns
import os
from scipy.ndimage.interpolation import rotate as scipy_rotate

import matplotlib.pyplot as plt


class CellObjController(object):
    index = 0
    length = 0

    def __init__(self, cell_list):
        # data: Data object, image data should be 3d; z, row, column
        super(CellObjController, self).__init__()
        self.cell_list = cell_list
        self.length = len(self.cell_list)
        self.exclude_bools = np.zeros(self.length).astype(bool)
        self.nw = NavigationWindow()

        self.cell_window = MPLWindow(cell_list, parent=None)

        self.nw.first_button.clicked.connect(self._first)
        self.nw.prev_button.clicked.connect(self._prev)
        self.nw.next_button.clicked.connect(self._next)
        self.nw.current_frame_text.editingFinished.connect(self._frame_text)
        self.nw.last_button.clicked.connect(self._last)

        self.nw.keyPressed.connect(self.key_event_nw)

        self.nw.exclude_cb.clicked.connect(self._exclude_cb_checked)
        self.nw.done_button.clicked.connect(self._done)

        self.nw.closed.connect(self._nw_closed)

    def show(self):
        self.cell_window.show()
        self.nw.show()

    @pyqtSlot()
    def _nw_closed(self):
        self.cell_window.close()

    @pyqtSlot()
    def _done(self):
        self.nw.close()
        del self

    def _exclude_cb_checked(self):
        self.exclude_bools[self.index] = ~self.exclude_bools[self.index]

    def exclude(self):
        self.exclude_bools[self.index] = ~self.exclude_bools[self.index]
        self.nw.exclude_cb.setChecked(bool(self.exclude_bools[self.index]))

    def key_event_nw(self, event):
        if event.key() == QtCore.Qt.Key_Left:
            self._prev()
        elif event.key() == QtCore.Qt.Key_A:
            self._prev()
        elif event.key() == QtCore.Qt.Key_Right:
            self._next()
        elif event.key() == QtCore.Qt.Key_D:
            self._next()
        elif event.key() == QtCore.Qt.Key_E:
            self.exclude()

    def set_frame(self, i):
        if i >= self.length:
            self.index = self.length - 1
        elif i < 0:
            self.index = 0
        else:
            self.index = i

        self.nw.current_frame_text.setText(str(self.index))

        self.cell_window.update_figure(self.index)
        self.nw.exclude_cb.setChecked(bool(self.exclude_bools[self.index]))

    def _frame_text(self):
        i = int(self.nw.current_frame_text.text())
        self.set_frame(i)

    @pyqtSlot()
    def _first(self):
        self.set_frame(0)

    @pyqtSlot()
    def _prev(self):
        self.set_frame(self.index - 1)

    @pyqtSlot()
    def _next(self):
        self.set_frame(self.index + 1)

    @pyqtSlot()
    def _last(self):
        self.set_frame(self.length - 1)
