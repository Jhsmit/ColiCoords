from images_select import NavigationWindow, ImageWindow
from preprocess_gui import InputWindow
from cell_objects import CellObjectWindow
from ..data import Data
from PyQt4 import QtCore
import numpy as np
import os
import tifffile


def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]


class InputController(object):

    def __init__(self):
        self.iw = InputWindow()
        self.iw.image_filter_button.clicked.connect(self._launch_image_filter)

    def show(self):
        self.iw.show()

    def _launch_image_filter(self):
        self.data = self._prepare_data()
        self.ctrl = ImageSelectController(self.data, self.iw.output_path)

    def _launch_image_filter_dep(self):
        data_dict = {}
        list_len = None
        for i in range(self.iw.input_list.count()):
            #todo make common function for this with universal data type dict format
            # -> put it in one of those handy data classes we have lying aroudn
            item = self.iw.input_list.item(i)
            w = self.iw.input_list.itemWidget(item)
            assert w.path

            file_list = listdir_fullpath(w.path)
            if list_len:
                assert len(file_list) == list_len
            list_len = len(file_list)
            shape = tifffile.imread(file_list[0]).shape
            data_arr = np.empty((len(file_list), shape[0], shape[1])).astype('uint16')

            for idx, f in enumerate(file_list):
                data_arr[idx] = tifffile.imread(f)

            name = w.name_lineedit.text()
            assert name is not None
            data_dict[name] = data_arr

        assert self.iw.output_path
        self.ctrl = ImageSelectController(data_dict, list_len, self.iw.output_path) #todo do something with this controller?

    def _launch_cell_objects(self):
        self.data = self._prepare_data()

    def _prepare_data(self):
        data = Data()
        list_len = None
        for i in range(self.iw.input_list.count()):
            # todo make common function for this with universal data type dict format
            # -> put it in one of those handy data classes we have lying aroudn
            item = self.iw.input_list.item(i)
            w = self.iw.input_list.itemWidget(item)
            assert w.path

            file_list = listdir_fullpath(w.path)
            if list_len:
                assert len(file_list) == list_len
            list_len = len(file_list)
            shape = tifffile.imread(file_list[0]).shape
            data_arr = np.empty((len(file_list), shape[0], shape[1])).astype('uint16')

            for idx, f in enumerate(file_list):
                data_arr[idx] = tifffile.imread(f)

            name = w.name_lineedit.text()
            assert name
            dclass = w.dclass_combobox.currentText()

            data.add_data(data_arr, dclass, name=name)

        return data


class ImageSelectController(object):
    index = 0
    length = 0

    def __init__(self, data, output_path):
        # data: Data object, image data should be 3d; z, row, column
        super(ImageSelectController, self).__init__()
        self.data = data
        self.length = len(self.data.data_dict.values()[0])
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

        for iw in self.iws:
            iw.show()
        self.nw.show()

    def _nw_closed(self):
        for iw in self.iws:
            iw.close()

    def _done(self):
        for name, data in self.data_dict.items():
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


class CellObjectController(object):
    def __init__(self, data):
        super(CellObjectController, self).__init__()

        self.cow = CellObjectWindow(data)

    def show(self):
        self.cow.show()