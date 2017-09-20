from cellcoordinates.gui.images_select import NavigationWindow, ImageWindow
import sys
from cellcoordinates.gui.preprocess_gui import InputWindow
from ..config import cfg
from cellcoordinates.gui.cell_objects import CellObjectWindow
from ..data_models import Data
from ..cell import Cell, CellList
from ..fileIO import save, load
from PyQt4 import QtCore, QtGui
import mahotas as mh
import numpy as np
import os
import tifffile
import math
import seaborn as sns
import os
from scipy.ndimage.interpolation import rotate as scipy_rotate

import matplotlib.pyplot as plt


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


class CellObjectController(object):
    def __init__(self, data, output_path):
        super(CellObjectController, self).__init__()
        self.input_data = data
        self.output_path = output_path

        if QtGui.QApplication.instance() is not None:
            self.cow = CellObjectWindow(data)
        else:
            self.cow = None

        self.cow.done_button.clicked.connect(self._done)

    def show(self):
        self.cow.show()

    def _done(self):
        cell_frac = float(self.cow.max_fraction_le.text())
        pad_width = int(self.cow.pad_width_le.text())
        rotate = self.cow.rotate_cbb.currentText()
        self.cell_list = self._create_cell_objects(self.input_data, cell_frac, pad_width, rotate)

        data_src = self.cow.optimize_datasrc_cbb.currentText()
        optimize_method = self.cow.optimize_method_cbb.currentText()

        self._optimize_coords(data_src, optimize_method)


        self._save_cellobjects()
        self._create_histograms()
       # self._save_metadata()


    def _create_cell_objects(self, input_data, cell_frac, pad_width, rotate):
        #todo move this function to preprocess and import
        cell_list = CellList()
        for i, data in enumerate(input_data):
            assert 'Binary' in data.dclasses

            #todo fix labeled binary in binary image!!!oneoneone
            binary = data.binary_img
            if (binary > 0).mean() > cell_frac or binary.mean() == 0.:
                print('Image {} {}: Too many or no cells').format(binary.name, i)
                continue

            # Iterate over all cells in the image
            for l in np.unique(binary)[1:]:
                selected_binary = (binary == l).astype('int')
                min1, max1, min2, max2 = mh.bbox(selected_binary)
                min1p, max1p, min2p, max2p = min1 - pad_width, max1 + pad_width, min2 - pad_width, max2 + pad_width

                try:
                    assert min1p > 0 and min2p > 0 and max1p < data.shape[0] and max2p < data.shape[1]
                except AssertionError:
                    print('Cell {} on image {} {}: on the edge of the image'.format(l, binary.name, i))
                    continue
                try:
                    assert len(np.unique(binary[min1p:max1p, min2p:max2p])) == 2
                except AssertionError:
                    print('Cell {} on image {} {}: multiple cells per selection'.format(l, output_data.binary_img.name, i))
                    continue

                output_data = data[min1p:max1p, min2p:max2p]
                output_data.binary_img //= output_data.binary_img.max()

                # Calculate rotation angle and rotate selections
                if rotate:
                    r_data = output_data.data_dict[rotate]
                    assert r_data.ndim == 2
                    theta = _calc_orientation(r_data)
                else:
                    theta = 0

                rotated_data = output_data.rotate(theta)

                #Make cell object and add all the data
                #todo change cell initation and data adding interface
                c = Cell(data_obj=rotated_data)
                cell_list.append(c)

        return cell_list

    #staticmethods?
    def _optimize_coords(self, dclass=None, method='photons', verbose=True):
        #todo verbose option in GUI
        for c in self.cell_list:
            c.optimize(dclass=dclass, method=method, verbose=verbose)

    def _save_cellobjects(self):
        if self.cow.cell_obj_cb.isChecked():
            ext = self.cow.cell_obj_cbb.currentText()
            path = os.path.join(self.output_path, 'cell_objects')
            if not os.path.exists(path):
                os.mkdir(path)
            for i, c in enumerate(self.cell_list):
                name = 'No_label_' + str(i).zfill(3) if not c.label else c.label
                name += ext
                fullpath = os.path.join(path, name)

                save(fullpath, c)

    def _create_histograms(self):
        assert hasattr(self, 'cell_list')
        #Histograms of different properties of the cells via its coordinate system
        labels = np.array(['Radius', 'Length', 'Area', 'Volume'])
        units = np.array([r' ($\mu m$)', r' ($\mu m$)', r' ($\mu m^{2}$)', r' ($\mu m^{3}$ / fL)'])
        f_um = cfg.IMG_PIXELSIZE / 1000
        conv_f = np.array([f_um, f_um, f_um**2, f_um**3])
        cell_prop = np.array([cb.isChecked() for cb in self.cow.cell_prop_cbs])
        cell_prop_ascii = np.array([cb.isChecked() for cb in self.cow.cell_prop_ascii_cbs])

        if np.any(cell_prop):
            figure_out_path = os.path.join(self.output_path, 'figures')
            if not os.path.exists(figure_out_path):
                os.mkdir(figure_out_path)

        ascii_data = [self.cell_list.label]
        for l, u, f, bool_c, bool_a in zip(labels, units, conv_f, cell_prop, cell_prop_ascii):
            values = getattr(self.cell_list, l.lower())
            if bool_c:
                plt.figure()
                ax = sns.distplot(values * f, kde=False)
                ax.set_title('Cell ' + l)
                ax.set_ylabel('Cell count')
                ax.set_xlabel(l + u)
                plt.savefig(os.path.join(figure_out_path, l + '_dist.png'))
            if bool_a:
                ascii_data.append(values * f)

        if np.any(cell_prop_ascii):
            ascii_out_path = os.path.join(self.output_path, 'ascii')
            if not os.path.exists(ascii_out_path):
                os.mkdir(ascii_out_path)

            n_cols = np.sum(cell_prop_ascii)
            names = ['Cell label'] + list(labels[cell_prop_ascii]) #todo make this crap into a function for smitsuite
            widths = [12] + n_cols * [5]
            types = ['S12'] + n_cols * [float]
            fmt = '%5s' + n_cols * ' %5.2f'

            export_data = np.zeros(len(self.cell_list), dtype=[(n, t) for n, t in zip(names, types)])
            for data, name in zip(ascii_data, names):
                export_data[name] = data

            header = ' '.join([n.rjust(w, ' ') for n, w in zip(names, widths)])
            fullpath = os.path.join(ascii_out_path, 'cell_properties.txt')
            np.savetxt(fullpath, export_data, header=header, fmt=fmt)


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

    return storm_out  #ha ha