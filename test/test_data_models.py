from colicoords.data_models import BinaryImage, BrightFieldImage, FluorescenceImage, STORMTable, Data
from colicoords.fileIO import load_thunderstorm
from test.testcase import ArrayTestCase
from test.test_functions import load_testdata
from scipy.ndimage.interpolation import rotate as scipy_rotate
import os
import numpy as np
import unittest


class TestDataElements(ArrayTestCase):
    def test_binaryimage(self):
        testdata = np.round(np.random.rand(512, 512)).astype(int)
        binary_img = BinaryImage(testdata, name='test1234', metadata={'no_entries': 123})
        self.assertArrayEqual(testdata, binary_img)
        sl_binary = binary_img[20:100, 100:200]
        self.assertTrue(sl_binary.dclass == 'binary')
        self.assertTrue(sl_binary.name == 'test1234')

    def test_brightfieldimage(self):
        testdata = np.round(np.random.rand(512, 512)) * 2**16-1
        bf_img = BrightFieldImage(testdata, name='test1234', metadata={'no_entries': 123})
        sl_bf = bf_img[20:100, 100:200]
        self.assertTrue(sl_bf.dclass == 'brightfield')
        self.assertTrue(sl_bf.name == 'test1234')

    def test_fluorescence_img(self):
        testdata = np.round(np.random.rand(512, 512)) * 2**16-1
        fl_img = FluorescenceImage(testdata, name='test1234', metadata={'no_entries': 123})
        sl_fl = fl_img[20:100, 100:200]
        self.assertTrue(sl_fl.dclass == 'fluorescence')
        self.assertTrue(sl_fl.name == 'test1234')

    def test_fluorescence_mov(self):
        testdata = np.round(np.random.rand(512, 512, 10)) * 2**16-1
        fl_img = FluorescenceImage(testdata, name='test1234', metadata={'no_entries': 123})
        sl_fl = fl_img[:5, 20:100, 100:200]
        self.assertTrue(sl_fl.dclass == 'fluorescence')
        self.assertTrue(sl_fl.name == 'test1234')

    def test_data_class_storm(self):
        f_path = os.path.dirname(os.path.realpath(__file__))
        storm_data = load_thunderstorm(os.path.join(f_path, 'test_data/ds5/storm_table.csv'))
        storm_table = STORMTable(storm_data, name='test1234', metadata={'no_entries:': 123})
        storm_sl = storm_table[5: 20]
        self.assertTrue(storm_table.dclass == 'storm')
        self.assertTrue(storm_table.name == 'test1234')
        self.assertTrue(storm_sl.shape == (15,))


class TestMakeData(ArrayTestCase):
    def test_add_data(self):
        testdata_int = np.round(np.random.rand(512, 512)).astype(int)
        testdata_float = np.round(np.random.rand(512, 512)) * 2**16-1
        testdata_mov = np.round(np.random.rand(10, 512, 512)) * 2**16-1

        data = Data()

        with self.assertRaises(TypeError):  # Invalid dtype
            data.add_data(testdata_float, dclass='binary')

        data.add_data(testdata_int, dclass='binary')
        self.assertArrayEqual(testdata_int, data.data_dict['binary'])
        self.assertArrayEqual(testdata_int, data.binary_img)

        with self.assertRaises(ValueError):  # Invalid shape
            data.add_data(testdata_float.reshape(256, -1), 'fluorescence')

        with self.assertRaises(ValueError):  # Binary has to be unique
            data.add_data(testdata_int, dclass='binary', name='newbinaryname')

        data.add_data(testdata_float, dclass='brightfield')
        with self.assertRaises(ValueError):  # Same dclass data elements which will have the same name
            data.add_data(testdata_float, dclass='brightfield')

        self.assertEqual(testdata_float.shape, data.shape)

        data.add_data(testdata_mov, 'fluorescence', name='fluorescence_movie')


class TestData(ArrayTestCase):
    def setUp(self):
        self.data = load_testdata('ds4')

    def test_copying(self):
        data_copy = self.data.copy()
        for k, v in self.data.data_dict.items():
            self.assertArrayEqual(v, data_copy.data_dict[k])

        i = self.data.data_dict['fluorescence'][10, 10, 10]
        self.data.data_dict['fluorescence'][10, 10, 10] += 20
        self.assertEqual(self.data.data_dict['fluorescence'][10, 10, 10], i + 20)
        self.assertEqual(i, data_copy.data_dict['fluorescence'][10, 10, 10])

    def test_rotation(self):
        data_rotated = self.data[:2].rotate(60)
        rotated = scipy_rotate(self.data.binary_img[:2], -60, mode='nearest', axes=(-1, -2))
        self.assertArrayEqual(rotated, data_rotated.binary_img)
        self.assertEqual(len(data_rotated), 2)

    def test_iteration(self):
        for i, d in enumerate(self.data):
            with self.subTest(i=i):
                self.assertArrayEqual(self.data.binary_img[i], d.binary_img)

        self.assertEqual(len(self.data), 20)


if __name__ == '__main__':
    unittest.main()