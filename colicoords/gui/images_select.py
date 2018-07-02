from PyQt5 import QtGui
from PyQt5.QtWidgets import QPushButton, QMainWindow, QLineEdit, QHBoxLayout, QCheckBox, QVBoxLayout, QWidget, QSizePolicy
from PyQt5 import QtCore
import pyqtgraph as pg
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np
pg.setConfigOptions(imageAxisOrder='row-major')
from colicoords.plot import CellPlot

DCLASS_ORDER = {'binary': 0, 'brightfield': 1, 'fluorescence': 2, 'STOROM': 3}


class CellMplCanvas(FigureCanvas):
    def __init__(self, cell_list, parent=None, width=5, height=4):
        self.cell_list = cell_list
        no_axes = len(cell_list[0].data.data_dict)
        rows = int(np.floor(np.sqrt(no_axes)))
        cols = int(np.ceil(np.sqrt(no_axes)))
        fig, self.axes = plt.subplots(rows, cols, figsize=(width, height))
        #todo what if no_figures == 1

        dclasses = cell_list[0].data.dclasses
        dnames = np.array(cell_list[0].data.names)
        order = np.argsort([DCLASS_ORDER[dclass] for dclass in dclasses])
        self.axes_dict = {name: ax for name, ax in zip(dnames[order], self.axes.flatten())}

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        self.update_figure(0)
        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def update_figure(self, i):
        cp = CellPlot(self.cell_list[i])
        for name, ax in self.axes_dict.items():
            ax.clear()
            ax.set_title(name)
            cp.imshow(str(name), ax=ax)
            cp.plot_outline(ax=ax)

        plt.tight_layout()
        self.draw()


class MPLWindow(QMainWindow):
    def __init__(self, cell_list, parent=None, width=5, height=4):
        super(MPLWindow, self).__init__(parent)
        self.mpl_canvas = CellMplCanvas(cell_list, parent=self, width=width, height=height)
        self.setCentralWidget(self.mpl_canvas)

    def update_figure(self, i):
        self.mpl_canvas.update_figure(i)


class ImageWindow(QMainWindow):
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


class CellWindow(QMainWindow):
    def __init__(self, cell_list, parent=None, title='CellWindow'):
        super(CellWindow, self).__init__(parent)
        self.setWindowTitle(title)
        self.cell_list = cell_list


class NavigationWindow(QMainWindow):
    keyPressed = QtCore.pyqtSignal(QtGui.QKeyEvent)
    closed = QtCore.pyqtSignal()

    def __init__(self):
        super(NavigationWindow, self).__init__()
        self.setWindowTitle('Navigation')

        self.first_button = QPushButton('<<')
        self.prev_button = QPushButton('<')
        self.current_frame_text = QLineEdit()
        self.current_frame_text.setValidator(QtGui.QIntValidator())
        self.current_frame_text.setText('0')
        self.next_button = QPushButton('>')
        self.last_button = QPushButton('>>')

        hbox_top = QHBoxLayout()
        hbox_top.addWidget(self.first_button)
        hbox_top.addWidget(self.prev_button)
        hbox_top.addWidget(self.current_frame_text)
        hbox_top.addWidget(self.next_button)
        hbox_top.addWidget(self.last_button)

        hbox_bottom = QHBoxLayout()
        self.exclude_cb = QCheckBox('Exclude')
        self.done_button = QPushButton('Done!')

        hbox_bottom.addWidget(self.exclude_cb)
        hbox_bottom.addStretch(1)
        hbox_bottom.addWidget(self.done_button)

        vbox_overall = QVBoxLayout()
        vbox_overall.addLayout(hbox_top)
        vbox_overall.addLayout(hbox_bottom)

        w = QWidget()
        w.setLayout(vbox_overall)
        self.setCentralWidget(w)

    def closeEvent(self, event):
        self.closed.emit()
        event.accept()

    def keyPressEvent(self, event):
        super(NavigationWindow, self).keyPressEvent(event)
        self.keyPressed.emit(event)

