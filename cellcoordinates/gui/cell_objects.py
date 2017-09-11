from PyQt4 import QtGui, QtCore
from ..config import cfg

#todo data dict ->ordereddict?
#todo make controller load defaults after initialization
class CellObjectWindow(QtGui.QMainWindow):
    def __init__(self, data, parent=None):
        super(CellObjectWindow, self).__init__(parent=parent)
        self.data = data
        #bascially get all names of datasets which are not brighfield
        self.data_names = [d.name for d in data.data_dict.values() if d.dclass in ['Binary', 'Fluorescence', 'STORMTable']]
        left_vbox = QtGui.QVBoxLayout()
        left_vbox.addWidget(QtGui.QLabel('Options'))

        form1 = QtGui.QFormLayout()

        self.pad_width_le = QtGui.QLineEdit()
        self.pad_width_le.setValidator(QtGui.QIntValidator())
        self.pad_width_le.setText(str(cfg.PAD_WIDTH))
        form1.addRow(QtGui.QLabel('Pad width'), self.pad_width_le)

        self.max_fraction_le = QtGui.QLineEdit()
        self.max_fraction_le.setValidator(QtGui.QDoubleValidator(0., 1., 2))
        self.max_fraction_le.setText(str(cfg.CELL_FRACTION))
        form1.addRow(QtGui.QLabel('Max cell fraction'), self.max_fraction_le)

        self.rotate_cbb = QtGui.QComboBox()
        self.rotate_cbb.addItems(self.data_names + ['None']) #todo input this
        form1.addRow(QtGui.QLabel('Rotate'), self.rotate_cbb)

        left_vbox.addLayout(form1)
        left_vbox.addWidget(QtGui.QLabel('Optimization'))

        form2 = QtGui.QFormLayout()

        self.optimize_datasrc_cbb = QtGui.QComboBox()
        self.optimize_datasrc_cbb.addItems(self.data_names) # todo input this
        form2.addRow(QtGui.QLabel('Data source'), self.optimize_datasrc_cbb)

        self.optimize_method_cbb = QtGui.QComboBox()
        self.optimize_method_cbb.addItems(['Binary', 'Localizations', 'Photons'])
        #todo additems
        form2.addRow(QtGui.QLabel('Optimization method'), self.optimize_method_cbb)

        left_vbox.addLayout(form2)
        left_vbox.addStretch(1)

        #MIDDLE BOX#
        middle_vbox = QtGui.QVBoxLayout()
        middle_vbox.addWidget(QtGui.QLabel('Output'))

        form3 = QtGui.QFormLayout()

        self.cell_obj_cb = QtGui.QCheckBox('Cell objects')
        self.cell_obj_cb.setChecked(True)
        self.cell_obj_cbb = QtGui.QComboBox()
        self.cell_obj_cbb.addItems(['.cc', '.tif'])
        self.cell_obj_cbb.setCurrentIndex(0)
        form3.addRow(self.cell_obj_cb, self.cell_obj_cbb)

        #todo export len, area, volume to the same file
        self.cell_length_cb = QtGui.QCheckBox('Cell length')
        self.cell_length_cb.setChecked(True)
        self.cell_length_ascii_cb = QtGui.QCheckBox('ASCII')

        #todo put this in grpbox
        labels = ['Radius', 'Length', 'Area', 'Volume']
        self.cell_prop_cbs = [QtGui.QCheckBox(l) for l in labels]
        self.cell_prop_ascii_cbs = [QtGui.QCheckBox('ASCII') for l in labels]

        for cb1, cb2 in zip(self.cell_prop_cbs, self.cell_prop_ascii_cbs):
            form3.addRow(cb1, cb2)

        middle_vbox.addLayout(form3)
        middle_vbox.addStretch(1)

        #RIGHT BOX#
        right_vbox = QtGui.QVBoxLayout()
        right_vbox.addWidget(QtGui.QLabel('Distributions'))

        self.dist_list = ListWidget()

        right_vbox.addWidget(self.dist_list)

        hbox = QtGui.QHBoxLayout()
        add_btn = QtGui.QPushButton('Add')
        add_btn.clicked.connect(self._add_button_clicked)
        remove_btn = QtGui.QPushButton('Remove')
        remove_btn.clicked.connect(self._remove_button_clicked)
        hbox.addWidget(add_btn)
        hbox.addWidget(remove_btn)
        right_vbox.addLayout(hbox)

        final_hbox = QtGui.QHBoxLayout()
        final_hbox.addLayout(left_vbox)
        final_hbox.addLayout(middle_vbox)
        final_hbox.addLayout(right_vbox)

        w = QtGui.QWidget()
        w.setLayout(final_hbox)
        self.setCentralWidget(w)

    def _add_button_clicked(self):
        dnames = [d.name for d in self.data.data_dict.values() if d.dclass in ['Fluorescence', 'STORMTable']]
        widget = DistributionOutputQCustomWidget(dnames, parent=self)

        list_item = QtGui.QListWidgetItem(self.dist_list)
        list_item.setSizeHint(widget.sizeHint())
        self.dist_list.addItem(list_item)
        self.dist_list.setItemWidget(list_item, widget)

    def _remove_button_clicked(self):
        del_item = self.dist_list.takeItem(self.dist_list.currentRow())
        del del_item


class DistributionOutputQCustomWidget(QtGui.QWidget):
    def __init__(self, dnames, parent=None):
        super(DistributionOutputQCustomWidget, self).__init__(parent=parent)
        vbox = QtGui.QVBoxLayout()
        form1 = QtGui.QFormLayout()

        self.data_src_cbb = QtGui.QComboBox()
        self.data_src_cbb.addItems(dnames)
        form1.addRow(QtGui.QLabel('Data source:'), self.data_src_cbb)

        vbox.addLayout(form1)

        # gb = QtGui.QGroupBox('Radial distribution')
        # f_r = QtGui.QFormLayout()
        #
        # self.r_normal_cb = QtGui.QCheckBox()
        # self.r_normal_cb.setChecked(True)
        # self.r_normal_cb_ascii = QtGui.QCheckBox()
        # f_r.addRow(self.r_normal_cb, self.r_normal_cb_ascii)

        dists = ['r', 'l', 'alpha']
        dist_names = ['Radial distribution', 'Longitinudududial distribution', 'Angular distribution']
        for d, n in zip(dists, dist_names):
            gb = QtGui.QGroupBox(n)
            f = QtGui.QFormLayout()
            cb = QtGui.QCheckBox('Normal')
            setattr(self, d + '_normal_cb', cb)
            cb_ascii = QtGui.QCheckBox('ASCII')
            setattr(self, d + '_normal_ascii_cb', cb_ascii)
            f.addRow(cb, cb_ascii)

            cb_norm = QtGui.QCheckBox('Normalized')
            setattr(self, d + '_normalized_cb', cb_norm)
            cb_norm_ascii = QtGui.QCheckBox('ASCII')
            setattr(self, d + 'normalized_ascii_cb', cb_norm_ascii)
            f.addRow(cb_norm, cb_norm_ascii)

            gb.setLayout(f)
            vbox.addWidget(gb)

        self.setLayout(vbox)


class ListWidget(QtGui.QListWidget):
    def sizeHint(self):
        qs = QtCore.QSize()
        # QtGui.QScrollBar().sizeHint().width()
        qs.setWidth(DistributionOutputQCustomWidget(['']).sizeHint().width() + 34)
        qs.setHeight(DistributionOutputQCustomWidget(['']).sizeHint().height())
        return qs
