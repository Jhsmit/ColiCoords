from PyQt5 import QtGui
from PyQt5.QtWidgets import QPushButton, QMainWindow, QLineEdit, QHBoxLayout, QCheckBox, QVBoxLayout, QWidget, \
    QSizePolicy, QRadioButton, QFormLayout, QLabel, QGraphicsEllipseItem, QSlider
from PyQt5 import QtCore

import pyqtgraph as pg
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np
pg.setConfigOptions(imageAxisOrder='row-major')
from colicoords.plot import CellPlot

DCLASS_ORDER = {'binary': 0, 'brightfield': 1, 'fluorescence': 2, 'STORM': 3}


class CellMplCanvas(FigureCanvas):
    def __init__(self, cell_list, parent=None, width=5, height=4):
        self.cell_list = cell_list
        no_axes = len(cell_list[0].data.data_dict)
        cols = int(np.ceil(np.sqrt(no_axes)))
        rows = int(np.ceil(no_axes / cols))
        self.fig, self.axes = plt.subplots(rows, cols, figsize=(width, height))
        #todo what if no_figures == 1

        dclasses = cell_list[0].data.dclasses
        dnames = np.array(cell_list[0].data.names)
        order = np.argsort([DCLASS_ORDER[dclass] for dclass in dclasses])

        self.axes_dict = {name: ax for name, ax in zip(dnames[order], self.axes.flatten())}

        FigureCanvas.__init__(self, self.fig)
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

        self.fig.tight_layout()
        self.draw()
        plt.tight_layout()


class MPLWindow(QMainWindow):
    def __init__(self, cell_list, parent=None, width=5, height=4):
        super(MPLWindow, self).__init__(parent)
        self.mpl_canvas = CellMplCanvas(cell_list, parent=self, width=width, height=height)
        self.setCentralWidget(self.mpl_canvas)

    def update_figure(self, i):
        self.mpl_canvas.update_figure(i)


class DragImageItem(pg.ImageItem):
    sigMouseDrag = QtCore.pyqtSignal(object)
    drag_enabled = True

    def __init__(self, *args, **kwargs):
        super(DragImageItem, self).__init__(*args, **kwargs)

    def mouseDragEvent(self, ev):
        if self.drag_enabled:
            self.sigMouseDrag.emit(ev)
            return True
        else:
            super(DragImageItem, self).mouseDragEvent(ev)


class PaintOptionsWindow(QMainWindow):
    keypress = QtCore.pyqtSignal(QtGui.QKeyEvent)

    def __init__(self, *args, parent=None, title='PaintOptions', **kwargs):
        super(PaintOptionsWindow, self).__init__(parent, *args, **kwargs)
        self.setWindowTitle(title)
        self.brush_size_edit = QLineEdit()
        self.brush_size_edit.setValidator(QtGui.QIntValidator())

        vb = QVBoxLayout()
        self.paint_rb = QRadioButton('Paint')
        self.paint_rb.setChecked(True)
        self.zoom_rb = QRadioButton('Zoom')
        vb.addWidget(self.paint_rb)
        vb.addWidget(self.zoom_rb)

        self.alpha_slider = QSlider(QtCore.Qt.Horizontal)
        self.alpha_slider.setMinimum(0)
        self.alpha_slider.setMaximum(100)
        self.alpha_slider.setTickInterval(100)

        self.autosave_cb = QCheckBox()

        form = QFormLayout()
        form.addRow(QLabel('Brush size'), self.brush_size_edit)
        form.addRow(QLabel('Mouse mode'), vb)
        form.addRow(QLabel('Alpha'), self.alpha_slider)

        w = QWidget()
        w.setLayout(form)
        self.setCentralWidget(w)

    def keyPressEvent(self, event):
        self.keypress.emit(event)


class OverlayImageWindow(QMainWindow):  #todo mixin with ImageWindow
    alpha = 0.5
    keypress = QtCore.pyqtSignal(QtGui.QKeyEvent)

    def __init__(self, img_arr, binary_arr, parent=None, title='ImageWindow'):
        super(OverlayImageWindow, self).__init__(parent)
        self.setWindowTitle(title)
        self.img_arr = img_arr
        self.binary_arr = binary_arr

        win = pg.GraphicsLayoutWidget()
        self.vb = pg.ViewBox(enableMouse=False, enableMenu=False)

        self.img_item = DragImageItem(img_arr[0])

        pos = np.array([0., 1.])
        color = np.array([[0., 0., 0., 0.], [1., 0., 0., self.alpha]])
        cm = pg.ColorMap(pos, color)
        lut = cm.getLookupTable(0., 1.)

        self.overlay_item = pg.ImageItem(binary_arr[0])
        self.overlay_item.setLookupTable(lut)
        self.vb.addItem(self.img_item)
        self.vb.addItem(self.overlay_item)
        self.vb.setAspectLocked()
        win.addItem(self.vb)
        #
        self.circle = QGraphicsEllipseItem(30., 30., 0., 0.)
        #self.circle.setBrush(QtGui.QBrush(QtCore.Qt.yellow))
        self.vb.addItem(self.circle)

        hist = pg.HistogramLUTItem()
        hist.setImageItem(self.img_item)
        win.addItem(hist)

        win.setStyleSheet("background-color:black;")
        self.setCentralWidget(win)

    def make_lut(self):
        pos = np.array([0., 1.])
        color = np.array([[0., 0., 0., 0.], [1., 0., 0., self.alpha]])
        cm = pg.ColorMap(pos, color)
        lut = cm.getLookupTable(0., 1.)

        return lut

        #self.img_item.scene().sigMouseMoved.connect(self.mouseMoved)
        #
        #self.img_item.sigMouseDrag.connect(self.mouseDrag)
        #proxy = pg.SignalProxy(vb.scene().sigMouseMoved, rateLimit=60, slot=self.mouseMoved)

    # def mouseDrag(self, ev):
    #     if ev.isStart():
    #         pos = ev.buttonDownPos()
    #         print(pos)
    #         row, col = int(pos.y()), int(pos.x())
    #
    #         self.binary_arr[0][row, col] = 1
    #
    #     im = self.binary_to_rgb(self.binary_arr[0])
    #     self.overlay_item.setImage(im)
    #
    #     ev.accept()

    def keyPressEvent(self, event):
        self.keypress.emit(event)

    # def mouseMoved(self, evt):
    #
    #     # print(evt)
    #     #
    #     # print(self.img_item.mapFromScene(evt))
    #     scenePos = self.img_item.mapFromScene(evt)
    #     self.circle.setRect(scenePos.x() - 30/2, scenePos.y() - 30/2, 30, 30)
    #     # row, col = int(scenePos.y()), int(scenePos.x())
    #     # print(row, col)

    def set_frame(self, i):
        self.img_item.setImage(self.img_arr[i])
        self.overlay_item.setImage(self.binary_arr[i])


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



if __name__ == '__main__':
    import numpy as np


    bf = np.random.random((10, 512, 512))
    binary = np.zeros_like(bf)
    ow = OverlayImageWindow(bf, binary)

    print(ow)