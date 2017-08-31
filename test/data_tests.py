from cellcoordinates.data import BinaryImage, BrightFieldImage, FluorescenceImage, FluorescenceMovie, STORMTable, Data

import numpy as np
import unittest


class Test(unittest.TestCase):

    def test_binaryimage(self):
        testdata = np.round(np.random.rand(512, 512)).astype(int)
        binary_img = BinaryImage(testdata, label='test1234', metadata={'no_entries': 123})

    def test_brightfieldimage(self):
        testdata = np.round(np.random.rand(512, 512)) * 2**16-1
        bf_img = BrightFieldImage(testdata, label='test1234', metadata={'no_entries': 123})

    def test_fluorescence_img(self):
        testdata = np.round(np.random.rand(512, 512)) * 2**16-1
        fl_img = FluorescenceImage(testdata, label='test1234', metadata={'no_entries': 123})

#    todo test timing of this one
    def test_fluorescence_mov(self):
        testdata = np.round(np.random.rand(512, 512, 10)) * 2**16-1
        fl_img = FluorescenceMovie(testdata, label='test1234', metadata={'no_entries': 123})

    def test_data_class_storm(self):
        storm_data = np.genfromtxt('test_data/storm_table.csv', skip_header=1, delimiter=',')


if __name__ ==  '__main__':
    unittest.main()