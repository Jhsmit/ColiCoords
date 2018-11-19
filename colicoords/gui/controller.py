from colicoords.gui.images_select import NavigationWindow, MPLWindow, OverlayImageWindow, PaintOptionsWindow, ImageWindow
from PyQt5.QtCore import pyqtSlot
from PyQt5 import QtCore
import queue
import mahotas as mh
import numpy as np
import time
import tifffile
import os

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

    def _first(self):
        self.set_frame(0)

    def _prev(self):
        self.set_frame(self.index - 1)

    def _next(self):
        self.set_frame(self.index + 1)

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


class AutoSaveThread(QtCore.QThread):
    interval = 10
    terminate = False

    def __init__(self, binary_array, draw_thread, *args, **kwargs):
        self.binary_array = binary_array
        self.draw_thread = draw_thread

        super(AutoSaveThread, self).__init__(*args, **kwargs)

    def run(self):
        while not self.terminate:
            time.sleep(self.interval)

            self.draw_thread.queue.join()
            np.save('autosave.npy', self.binary_array)
            print('Autosaved')

        self.terminate = False
        return 0


class DrawThread(QtCore.QThread):
    brush_size = 10
    brush_size_sq = 100
    terminate = False

    def __init__(self, binary_array, image_window, *args, **kwargs):
        self.binary_array = binary_array
        self.iw = image_window
        self.edited = np.any(self.binary_array, axis=(1, 2)).astype(bool)

        super(DrawThread, self).__init__(*args, **kwargs)
        self.queue = queue.Queue()
        self.shape = self.binary_array[0].shape
        ymax = self.shape[0]
        xmax = self.shape[1]
        self.x_coords = np.repeat(np.arange(xmax), ymax).reshape(xmax, ymax).T + 0.5
        self.y_coords = np.repeat(np.arange(ymax), xmax).reshape(ymax, xmax) + 0.5
        self.zero = np.ones_like(binary_array[0], dtype=bool)

    def run(self):
        while not self.terminate:
            idx, x, y, value = self.queue.get()
            self.zero[int(y), int(x)] = False
            dmap = mh.distance(self.zero)
            bools = dmap < self.brush_size_sq
            self.binary_array[idx][bools] = value
            self.edited[idx] = True

            self.iw.overlay_item.setImage(self.binary_array[idx])
            self.zero[int(y), int(x)] = True

            self.queue.task_done()

        self.terminate = False
        return 0


class PropagateThread(QtCore.QThread):
    terminate = False

    def __init__(self, parent, *args, **kwargs):
        self.parent = parent
        super(PropagateThread, self).__init__(*args, **kwargs)
        self.edited = np.any(self.parent.binary_array, axis=(1, 2))

    def run(self):
        #todo trigger with next?
        while not self.terminate:
            print('running')

            binary = self.parent.binary_array[self.parent.index]
            idx = np.where(self.parent.draw_thread.edited)[0]
            try:
                i_final = np.min(idx[idx > self.parent.index])
            except ValueError:
                i_final = len(self.parent.binary_array)

            self.parent.binary_array[self.parent.index + 1:i_final, :, :] = binary[np.newaxis, :, :]
            time.sleep(5)


class GenerateBinaryController(NavigationMixin):

    def __init__(self, grey_array, binary_array, propagate=False):
        assert grey_array.shape == binary_array.shape
        self.grey_array = grey_array[:, ::-1, :]
        self.binary_array = binary_array[:, ::-1, :].astype(int)
        super(GenerateBinaryController, self).__init__(len(grey_array))

        self.shape = self.grey_array[0].shape

        #Binary overlay window
        self.iw = OverlayImageWindow(self.grey_array, self.binary_array)
        self.iw.img_item.sigMouseDrag.connect(self.on_mouse_drag)
        self.iw.keypress.connect(self.on_key_press)
        self.draw_thread = DrawThread(self.binary_array, self.iw)
        self.draw_thread.start()

        if propagate:
            self.prop = PropagateThread(self)
            self.prop.start()

        #self.autosave = AutoSaveThread(self.binary_array, self.draw_thread)
        #self.autosave.start()

        self.iw.img_item.scene().sigMouseMoved.connect(self.mouse_moved)

        #Paint options window
        self.pw = PaintOptionsWindow()
        self.update_brush_size_edit()
        self.pw.brush_size_edit.editingFinished.connect(self._brush_size_text)
        self.pw.paint_rb.toggled.connect(self._paint_mode_rb)
        self.pw.alpha_slider.setValue(self.iw.alpha * 100)
        self.pw.alpha_slider.valueChanged.connect(self.alpha_slider)
        self.pw.keypress.connect(self.on_key_press)

        self.nw.done_button.clicked.connect(self.on_done_button)

    def on_done_button(self):
        self.iw.update()
        self.iw.overlay_item.update()
        self.pw.update()
        self.set_frame(self.index)
        self._paint_mode_rb()

    def alpha_slider(self):
        self.iw.alpha = self.pw.alpha_slider.value() / 100
        lut = self.iw.make_lut()
        self.iw.overlay_item.setLookupTable(lut)

    def update_brush_size_edit(self):
        self.pw.brush_size_edit.setText(str(self.draw_thread.brush_size))
        self.draw_thread.brush_size_sq = int(self.draw_thread.brush_size**2)

    def _brush_size_text(self):
        brush_size = self.pw.brush_size_edit.text()
        self.draw_thread.brush_size = int(brush_size)
        self.draw_thread.brush_size_sq = int(self.draw_thread.brush_size**2)

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
            self.update_brush_size_edit()
        elif event.key() == QtCore.Qt.Key_R:
            self.draw_thread.brush_size -= 1
            self.update_brush_size_edit()

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


class ImageSelectController(object):
    index = 0
    length = 0

    def __init__(self, data, output_path):
        # data: Data object, image data should be 3d; z, row, column
        super(ImageSelectController, self).__init__()
        self.data = data
        self.length = len(self.data)
        self.output_path = output_path
        self.exclude_bools = np.zeros(self.length).astype(bool)
        self.nw = NavigationWindow()
        self.iws = []
        for k, v in self.data.data_dict.items():
            iw = ImageWindow(v, parent=None, title=k)
            self.iws.append(iw)
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
        for iw in self.iws:
            iw.show()
        self.nw.show()

    def _nw_closed(self):
        for iw in self.iws:
            iw.close()

    def _done(self):
        for name, data in self.data.data_dict.items():
            name = str(name) #Otherwise QString
            export_data = data[~self.exclude_bools]
            out_dir = os.path.join(self.output_path, name)
            if not os.path.exists(out_dir):
                os.mkdir(out_dir)
            for i, d in enumerate(export_data):
                tifffile.imsave(os.path.join(out_dir, name + '_' + str(i).zfill(3)) + '.tif', d)
        for w in self.iws:
            w.close()
        self.nw.close()
        del self #NUKE em
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
        for iw in self.iws:
            iw.set_frame(self.index)
        self.nw.exclude_cb.setChecked(bool(self.exclude_bools[self.index]))
    def _frame_text(self):
        i = int(self.nw.current_frame_text.text())
        self.set_frame(i)
    def _first(self):
        self.set_frame(0)
    def _prev(self):
        self.set_frame(self.index - 1)
    def _next(self):
        self.set_frame(self.index + 1)
    def _last(self):
        self.set_frame(self.length - 1)