from images_select import NavigationWindow, ImageWindow
from preprocess_gui import InputWindow
from PyQt4 import QtCore
import numpy as np
import os
import tifffile


def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]


class InputController(object):
    output_path = ''

    def __init__(self):
        self.iw = InputWindow()
        self.iw.image_filter_button.clicked.connect(self._launch_image_filter)
        self.iw.show()

    def _launch_image_filter(self):
        data_dict = {}
        list_len = None
        for i in range(self.iw.input_list.count()):

            item = self.iw.input_list.item(i)
            w = self.iw.input_list.itemWidget(item)
            assert w.path is not None

            file_list = listdir_fullpath(w.path)
            if list_len:
                assert len(file_list) == list_len
            list_len = len(file_list)
            shape = tifffile.imread(file_list[0]).shape
            data_arr = np.empty((len(file_list), shape[0], shape[1]))

            for idx, f in enumerate(file_list):
                data_arr[idx] = tifffile.imread(f)

            name = w.name_lineedit.text()
            assert name is not None
            data_dict[name] = data_arr

        self.output_path = self.iw.output_path
        assert self.output_path is not None
        self.ctrl = ImageSelectController(data_dict, list_len, self.output_path) #todo do something with this controller?



class ImageSelectController(object):
    index = 0
    length = 0

    def __init__(self, data_dict, length, output_path):
        # data dict: k; name of the data, v; 3d array (z, x, y)
        super(ImageSelectController, self).__init__()
        self.length = length
        self.output_path = output_path
        self.data_dict = data_dict
        self.exclude_bools = np.zeros(self.length).astype(bool)

        self.nw = NavigationWindow()

        self.iws = []
        for k, v in data_dict.items():
            iw = ImageWindow(v, parent=self.nw, title=k)
            self.iws.append(iw)

        self.nw.first_button.clicked.connect(self._first)
        self.nw.prev_button.clicked.connect(self._prev)
        self.nw.next_button.clicked.connect(self._next)
        self.nw.current_frame_text.editingFinished.connect(self._frame_text)
        self.nw.last_button.clicked.connect(self._last)

        self.nw.keyPressed.connect(self.key_event_nw)

        self.nw.exclude_cb.clicked.connect(self._exclude_cb_checked)
        self.nw.done_button.clicked.connect(self._done)

        for iw in self.iws:
            iw.show()
        self.nw.show()

    def set_frame(self, i):
        self.index = i
        self.index = self.index % self.length

        self.nw.current_frame_text.setText(str(self.index))

        for iw in self.iws:
            iw.set_frame(self.index)
        self.nw.exclude_cb.setChecked(bool(self.exclude_bools[self.index]))

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