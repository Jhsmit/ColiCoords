from images_select import NavigationWindow, ImageWindow
from preprocess_gui import InputWindow
from ..config import cfg
from cell_objects import CellObjectWindow
from ..data import Data
from ..cell import Cell
from ..fileIO import save, load
from PyQt4 import QtCore, QtGui
import mahotas as mh
import numpy as np
import os
import tifffile
import math
from scipy.ndimage.interpolation import rotate as scipy_rotate


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
        self.ctrl.show()

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
        self.ctrl = CellObjectController(self.data, self.iw.output_path)
        self.ctrl.show()

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

            name = w.name_lineedit.text() # can be None
            dclass = w.dclass_combobox.currentText()
            data.add_data(data_arr, str(dclass), name=name)

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
    def __init__(self, data, output_path):
        super(CellObjectController, self).__init__()
        self.input_data = data
        self.output_path = output_path

        if QtGui.QApplication.instance() is not None:
            self.cow = CellObjectWindow(data)
        else:
            self.cow = None

    def show(self):
        self.cow.show()

    def _done(self):
        cell_frac = float(self.cow.max_fraction_le.text())
        pad_width = int(self.cow.pad_width_le.text())
        rotate = self.cow.rotate_cbb.currentText()
        cell_list = self._create_cell_objects(self.input_data, cell_frac, pad_width, rotate)

        data_src = self.cow.optimize_datasrc_cbb.currentText()
        optimize_method = self.cow.optimize_method_cbb.currentText()

        self._optimize_coords(cell_list, data_src, optimize_method)
        self._create_output(cell_list)
        #purrhaps cell_list should be an attribute

    def _create_cell_objects(self, input_data, cell_frac, pad_width, rotate):
        cell_list = []
        for i, data in enumerate(input_data):

            assert 'Binary' in data.dclasses
            binary = data.binary_img
            if (binary > 0).mean() > cell_frac or binary.mean() == 0.:
                print('Image {} {}: Too many or no cells').format(binary.name, i)

            #Iterate over all cells in the image
            for l in np.unique(binary)[1:]:
                selected_binary = (binary == l).astype('int')
                min1, max1, min2, max2 = mh.bbox(selected_binary)
                min1p, max1p, min2p, max2p = min1 - pad_width, max1 + pad_width, min2 - pad_width, max2 + pad_width
                bin_selection = selected_binary[min1p:max1p, min2p:max2p]

                try:
                    assert min1p > 0 and min2p > 0 and max1p < binary.shape[0] and max2p < binary.shape[1]
                except AssertionError:
                    print('Cell {} on image {} {}: on the edge of the image'.format(l, binary.name, i))
                    continue

                try:
                    assert len(np.unique(bin_selection)) == 2
                except AssertionError:
                    print('Cell {} on image {} {}: multiple cells per selection'.format(l, binary.name, i))
                    continue

            bin_selection = bin_selection.astype(int)  # Otherwise the rotation result looks funny

            flu_selection = None
            if data.flu_dict:
                flu_selection = {}
                for k, v in data.flu_dict.items():
                    flu_selection[k] = v[min1 - pad_width:max1 + pad_width, min2 - pad_width:max2 + pad_width]

            bf_selection = data.brightfield_img[min1 - pad_width:max1 + pad_width,
                           min2 - pad_width:max2 + pad_width] if data.brightfield_img is not None else None

            if data.storm_table:
                raise NotImplementedError('Handling of STORM data not implemented')

            # Calculate rotation angle and rotate selections
            if rotate:
                #assert (-> get by name)
                r_data = data.data_dict[rotate]
                print(r_data.dclass)
                assert r_data.ndim == 2
                theta = _calc_orientation(r_data)
            else:
                theta = 0
            import matplotlib.pyplot as plt
            # plt.imshow(bin_selection)
            # plt.show()


            bin_rotated = scipy_rotate(bin_selection, theta)  #todo change if its ever allowed not to have binary
            bf_rotated = scipy_rotate(bf_selection, theta) if bf_selection is not None else None

            plt.imshow(bin_rotated)
            plt.show()

            flu_rotated = {}
            for k, v in flu_selection.items():
                flu_rotated[k] = scipy_rotate(v, theta)

            #Make cell object and add all the data
            #todo change cell initation and data adding interface
            c = Cell(bf_img=bf_rotated, binary_img=bin_rotated, flu_data=flu_rotated, storm_table=None)
            cell_list.append(c)

        return cell_list

    #staticmethods?
    def _optimize_coords(self, cell_list, dclass=None, method='photons', verbose=True):
        #todo verbose option in GUI
        for c in cell_list:
            c.optimize(dclass=dclass, method=method, verbose=verbose)

    def _create_output(self, cell_list):
        if self.cow.cell_obj_cb.isChecked():
            ext = self.cow.cell_obj_cbb.currentText()
            path = os.path.join(self.output_path, 'cell_objects')
            if not os.path.exists(path):
                os.mkdir(path)
            self._save_cellobjects(cell_list, path, ext)

    def _save_cellobjects(self, cell_list, path, ext):
        for c in cell_list:
            name = os.path.join(path, c.label + ext)
            save(name, c)

    def _create_histograms(self):
        pass


def _calc_orientation(data_elem):
    if data_elem.dclass in ['Binary', 'Fluorescence']:
        img = data_elem
    elif data_elem.dclass == 'STORMTable':
        xmax = int(data_elem['x'].max()) + 2 * cfg.STORM_PIXELSIZE
        ymax = int(data_elem['y'].max()) + 2 * cfg.STORM_PIXELSIZE
        x_bins = np.arange(0, xmax, cfg.STORM_PIXELSIZE)
        y_bins = np.arange(0, ymax, cfg.STORM_PIXELSIZE)

        img, xedges, yedges = np.histogram2d(data_elem['x'], data_elem['y'], bins=[x_bins, y_bins])

    else:
        raise ValueError('Invalid dtype')

    com = mh.center_of_mass(img)

    mu00 = mh.moments(img, 0, 0, com)
    mu11 = mh.moments(img, 1, 1, com)
    mu20 = mh.moments(img, 2, 0, com)
    mu02 = mh.moments(img, 0, 2, com)

    mup_20 = mu20 / mu00
    mup_02 = mu02 / mu00
    mup_11 = mu11 / mu00

    theta_rad = 0.5 * math.atan(2 * mup_11 / (mup_20 - mup_02))  # todo math -> numpy
    theta = theta_rad * (180 / math.pi)
    if (mup_20 - mup_02) > 0:
        theta += 90

    return theta


def _rotate_storm(storm_data, theta, shape=None):
    theta *= np.pi / 180  # to radians
    x = storm_data['x']
    y = storm_data['y']

    if shape:
        xmax = shape[0] * cfg.IMG_PIXELSIZE
        ymax = shape[1] * cfg.IMG_PIXELSIZE
    else:
        xmax = int(storm_data['x'].max()) + 2 * cfg.STORM_PIXELSIZE
        ymax = int(storm_data['y'].max()) + 2 * cfg.STORM_PIXELSIZE

    x -= xmax / 2
    y -= ymax / 2

    xr = x * np.cos(theta) + y * np.sin(theta)
    yr = y * np.cos(theta) - x * np.sin(theta)

    xr += xmax / 2
    yr += ymax / 2

    storm_out = np.copy(storm_data)
    storm_out['x'] = xr
    storm_out['y'] = yr

    return storm_out