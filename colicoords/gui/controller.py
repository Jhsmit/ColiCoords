from colicoords.gui.images_select import NavigationWindow, MPLWindow, OverlayImageWindow, PaintOptionsWindow
from PyQt5.QtCore import pyqtSlot
from PyQt5 import QtCore
import queue
import mahotas as mh
import numpy as np


class NavigationMixin(object):
    index = 0
    length = 0

    def __init__(self, length):
        self.length = length
        self.nw = NavigationWindow()

        self.nw.first_button.clicked.connect(self._first)
        self.nw.prev_button.clicked.connect(self._prev)
        self.nw.next_button.clicked.connect(self._next)
        self.nw.current_frame_text.editingFinished.connect(self._frame_text)
        self.nw.last_button.clicked.connect(self._last)
        self.nw.keyPressed.connect(self.on_key_press)

        #self.nw.closed.connect(self._nw_closed)

    def on_key_press(self, event):
        if event.key() == QtCore.Qt.Key_Left:
            self._prev()
        elif event.key() == QtCore.Qt.Key_A:
            self._prev()
        elif event.key() == QtCore.Qt.Key_Right:
            self._next()
        elif event.key() == QtCore.Qt.Key_D:
            self._next()

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

    def _frame_text(self):
        i = int(self.nw.current_frame_text.text())
        self.set_frame(i)

    def set_frame(self, i):
        if i >= self.length:
            self.index = self.length - 1
        elif i < 0:
            self.index = 0
        else:
            self.index = i

        self.nw.current_frame_text.setText(str(self.index))


class DrawThread(QtCore.QThread):
    brush_size = 10

    def __init__(self, binary_array, image_window, *args, **kwargs):
        self.binary_array = binary_array
        self.iw = image_window

        super(DrawThread, self).__init__(*args, **kwargs)
        self.queue = queue.Queue()
        self.shape = self.binary_array[0].shape
        ymax = self.shape[0]
        xmax = self.shape[1]
        self.x_coords = np.repeat(np.arange(xmax), ymax).reshape(xmax, ymax).T + 0.5
        self.y_coords = np.repeat(np.arange(ymax), xmax).reshape(ymax, xmax) + 0.5
        self.zero = np.ones_like(binary_array[0], dtype=bool)
        self.terminate = False

    def run(self):
        while not self.terminate:
            idx, x, y, value = self.queue.get()
            self.zero[int(y), int(x)] = False
            dmap = mh.distance(self.zero)

            bools = dmap < self.brush_size**2
            self.binary_array[idx][bools] = value

            self.iw.overlay_item.setImage(self.binary_array[idx])
            self.zero[int(y), int(x)] = True

            self.queue.task_done()


class GenerateBinaryController(NavigationMixin):

    def __init__(self, grey_array, binary_array):
        assert grey_array.shape == binary_array.shape
        self.grey_array = grey_array[:, ::-1, :]
        self.binary_array = binary_array[:, ::-1, :]
        super(GenerateBinaryController, self).__init__(len(grey_array))

        self.shape = self.grey_array[0].shape

        #Binary overlay window
        self.iw = OverlayImageWindow(self.grey_array, self.binary_array)
        self.iw.img_item.sigMouseDrag.connect(self.on_mouse_drag)
        self.iw.keypress.connect(self.on_key_press)
        self.draw_thread = DrawThread(self.binary_array, self.iw)
        self.draw_thread.start()

        self.iw.img_item.scene().sigMouseMoved.connect(self.mouse_moved)

        #Paint options window
        self.pw = PaintOptionsWindow()
        self.update_brush_size_edit()
        self.pw.brush_size_edit.editingFinished.connect(self._brush_size_text)
        self.pw.paint_rb.toggled.connect(self._paint_mode_rb)
        self.pw.keypress.connect(self.on_key_press)

    def update_brush_size_edit(self):
        self.pw.brush_size_edit.setText(str(self.draw_thread.brush_size))

    def mouse_moved(self, ev):
        scenePos = self.iw.img_item.mapFromScene(ev)
        r = self.draw_thread.brush_size
        self.iw.circle.setRect(scenePos.x() - r, scenePos.y() - r, 2*r, 2*r)

    def on_key_press(self, event):
        if event.key() == QtCore.Qt.Key_Left:
            self._prev()
        elif event.key() == QtCore.Qt.Key_A:
            self._prev()
        elif event.key() == QtCore.Qt.Key_Right:
            self._next()
        elif event.key() == QtCore.Qt.Key_D:
            self._next()
        elif event.key() == QtCore.Qt.Key_F:
            bool = self.pw.paint_rb.isChecked()

            if bool:
                self.pw.zoom_rb.toggle()
            else:
                self.pw.paint_rb.toggle()
        elif event.key() == QtCore.Qt.Key_E:
            self.draw_thread.brush_size += 1
            print(self.draw_thread.brush_size)
            self.update_brush_size_edit()
        elif event.key() == QtCore.Qt.Key_R:
            self.draw_thread.brush_size -= 1
            self.update_brush_size_edit()

    def _brush_size_text(self):
        brush_size = self.pw.brush_size_edit.text()
        self.draw_thread.brush_size = int(brush_size)

    def _paint_mode_rb(self):
        bool = self.pw.paint_rb.isChecked()
        if bool:  # enabling paint mode, disable mouse mode
            self.iw.img_item.drag_enabled = True
            self.iw.vb.setMouseEnabled(x=False, y=False)
            self.iw.vb.setMenuEnabled(False)
        else:
            self.iw.img_item.drag_enabled = False
            self.iw.vb.setMouseEnabled(x=True, y=True)
            self.iw.vb.setMenuEnabled(True)

    def show(self):
        self.nw.show()
        self.iw.show()
        self.pw.show()

    def set_frame(self, i):
        super(GenerateBinaryController, self).set_frame(i)  # not sure if this is 100% correct and will still work with multiple mixins
        self.iw.set_frame(self.index)

    def on_mouse_drag(self, ev):
        if ev.isStart():
            pos = ev.buttonDownPos()
            y, x = int(pos.y()), int(pos.x())

        else:
            pos = ev.pos()
            y, x = int(pos.y()), int(pos.x())

        x = np.min([np.max([0, x]), self.shape[1] - 1])
        y = np.min([np.max([0, y]), self.shape[0] - 1])

        if ev.button() == 4:
            ev.ignore()
        else:

            value = 2 - ev.button()
            self.draw_thread.queue.put((self.index, x, y, value))
            ev.accept()


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
