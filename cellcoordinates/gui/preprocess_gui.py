from PyQt4 import QtGui, QtCore
import pyqtgraph as pg
import matplotlib.pyplot as plt
import sys
import numpy as np
pg.setConfigOptions(imageAxisOrder='row-major')

#todo allow input of img stack as input data as well

#https://blog.manash.me/quick-qt-3-how-to-dynamically-create-qlistwidgetitem-and-add-it-onto-qlistwidget-4bca5bacaa01
class InputWindow(QtGui.QMainWindow): # todo could use some renaming at some point
    output_path = ''

    def __init__(self):
        super(InputWindow, self).__init__()

        #Left Column, input
        self.input_list = ListWidget()
     #   print(DataInputQCustomWidget().sizeHint().width())
       # self.input_list.sizeHintForColumn(DataInputQCustomWidget().sizeHint().width())
#
        add_button = QtGui.QPushButton('Add')
        add_button.clicked.connect(self._add_button_clicked)

        remove_button = QtGui.QPushButton('Remove')
        remove_button.clicked.connect(self._remove_button_clicked)

        vbox_input = QtGui.QVBoxLayout()
        vbox_input.addWidget(QtGui.QLabel('Input data:'))
        vbox_input.addWidget(self.input_list)
        #vbox_input.addStretch(1)

        hbox = QtGui.QHBoxLayout()
        hbox.addWidget(add_button)
        hbox.addWidget(remove_button)

        vbox_input.addLayout(hbox)

        #Middle column, output
        vbox_output = QtGui.QVBoxLayout()
        vbox_output.addWidget(QtGui.QLabel('Output:'))

        output_form = QtGui.QFormLayout()
        output_path_btn = QtGui.QPushButton("Browse")
        output_path_btn.clicked.connect(self._output_path_btn_clicked)

        output_form.addRow(QtGui.QLabel('Path:'), output_path_btn)
        vbox_output.addLayout(output_form)
        vbox_output.addStretch(1)

        vbox_processing = QtGui.QVBoxLayout()
        vbox_processing.addWidget(QtGui.QLabel('Processing'))

        self.image_filter_button = QtGui.QPushButton('Filter Images')
        vbox_processing.addWidget(self.image_filter_button)

        vbox_processing.addStretch(1)

        hbox = QtGui.QHBoxLayout()
        hbox.addLayout(vbox_input)
        hbox.addLayout(vbox_output)
        hbox.addLayout(vbox_processing)

        w = QtGui.QWidget()
        w.setLayout(hbox)
        self.setCentralWidget(w)

    def _output_path_btn_clicked(self):
        self.output_path = str(QtGui.QFileDialog.getExistingDirectory(self, "Select Directory", self.output_path))

    def _add_button_clicked(self):
        self._add_list_item()

    def _add_list_item(self):
        widget = DataInputQCustomWidget(self)
        list_item = QtGui.QListWidgetItem(self.input_list)
        list_item.setSizeHint(widget.sizeHint())
        self.input_list.addItem(list_item)
        self.input_list.setItemWidget(list_item, widget)

    def _remove_button_clicked(self):
        del_item = self.input_list.takeItem(self.input_list.currentRow())
        del del_item

    # def _add_table_row(self, name):
    #     label = QtGui.QLabel(name + str(self.input_table.rowCount()))
    #     btn = QtGui.QPushButton('Remove')
    #     btn._row = self.input_table.rowCount()
    #   #  btn.clicked.connect(self._remove_button_clicked)
    #     self.input_table.insertRow(self.input_table.rowCount())
    #
    #     self.input_table.setCellWidget(self.input_table.rowCount() - 1, 0, label)
    #     self.input_table.setCellWidget(self.input_table.rowCount() - 1, 1, btn)
    #     self.input_table.resizeRowsToContents()


class ListWidget(QtGui.QListWidget):
    def sizeHint(self):
        qs = QtCore.QSize()
        # QtGui.QScrollBar().sizeHint().width()
        qs.setWidth(DataInputQCustomWidget().sizeHint().width() + 34)
        qs.setHeight(3*DataInputQCustomWidget().sizeHint().height())
        return qs


class DataInputQCustomWidget(QtGui.QWidget):
    path = ''  #todo choose default dir in config

    def __init__(self, parent=None):
        super(DataInputQCustomWidget, self).__init__(parent)
        form = QtGui.QFormLayout()

        self.name_lineedit = QtGui.QLineEdit()
        form.addRow(QtGui.QLabel('Name:'), self.name_lineedit)

        self.dclass_combobox = QtGui.QComboBox()
        self.dclass_combobox.addItems(['Binary', 'Brightfield', 'Fluorescence', 'STORM', ])
        form.addRow(QtGui.QLabel('Data Type:'), self.dclass_combobox)

        self.path_button = QtGui.QPushButton("Browse")
        self.path_button.clicked.connect(self.path_button_clicked)
        form.addRow(QtGui.QLabel('Path:'), self.path_button)

        self.setLayout(form)


    def path_button_clicked(self):
        self.path = str(QtGui.QFileDialog.getExistingDirectory(self, "Select Directory", self.path))


class AddDataDialog(QtGui.QDialog):
    def __init__(self):
        super(AddDataDialog, self).__init__()


