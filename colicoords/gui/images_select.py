from PyQt4 import QtGui, QtCore
import pyqtgraph as pg
import matplotlib.pyplot as plt
import sys
import numpy as np
pg.setConfigOptions(imageAxisOrder='row-major')


class ImageWindow(QtGui.QMainWindow):
    def __init__(self, img_arr, parent=None, title='ImageWindow'):
        super(ImageWindow, self).__init__(parent)
        self.setWindowTitle(title)
        self.img_arr = img_arr
        win = pg.GraphicsLayoutWidget()
        vb = pg.ViewBox(enableMouse=False, enableMenu=False)

        self.img_item = pg.ImageItem(img_arr[0])
        vb.addItem(self.img_item)
        vb.setAspectLocked()
        win.addItem(vb)

        hist = pg.HistogramLUTItem()
        hist.setImageItem(self.img_item)
        win.addItem(hist)

        win.setStyleSheet("background-color:black;")
        self.setCentralWidget(win)

    def set_frame(self, i):
        self.img_item.setImage(self.img_arr[i])


class NavigationWindow(QtGui.QMainWindow):
    keyPressed = QtCore.pyqtSignal(QtGui.QKeyEvent)
    closed = QtCore.pyqtSignal()

    def __init__(self):
        super(NavigationWindow, self).__init__()
        self.setWindowTitle('Navigation')

        self.first_button = QtGui.QPushButton('<<')
        self.prev_button = QtGui.QPushButton('<')
        self.current_frame_text = QtGui.QLineEdit()
        self.current_frame_text.setValidator(QtGui.QIntValidator())
        self.current_frame_text.setText('0')
        self.next_button = QtGui.QPushButton('>')
        self.last_button = QtGui.QPushButton('>>')

        hbox_top = QtGui.QHBoxLayout()
        hbox_top.addWidget(self.first_button)
        hbox_top.addWidget(self.prev_button)
        hbox_top.addWidget(self.current_frame_text)
        hbox_top.addWidget(self.next_button)
        hbox_top.addWidget(self.last_button)

        hbox_bottom = QtGui.QHBoxLayout()
        self.exclude_cb = QtGui.QCheckBox('Exclude')
        self.done_button = QtGui.QPushButton('Done!')

        hbox_bottom.addWidget(self.exclude_cb)
        hbox_bottom.addStretch(1)
        hbox_bottom.addWidget(self.done_button)

        vbox_overall = QtGui.QVBoxLayout()
        vbox_overall.addLayout(hbox_top)
        vbox_overall.addLayout(hbox_bottom)

        w = QtGui.QWidget()
        w.setLayout(vbox_overall)
        self.setCentralWidget(w)

    def closeEvent(self, event):
        self.closed.emit()
        event.accept()

    def keyPressEvent(self, event):
        super(NavigationWindow, self).keyPressEvent(event)
        self.keyPressed.emit(event)

        #todo override close event


