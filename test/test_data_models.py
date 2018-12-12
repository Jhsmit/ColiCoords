from colicoords.data_models import BinaryImage, BrightFieldImage, FluorescenceImage, STORMTable, Data
from colicoords.fileIO import load_thunderstorm, load
from colicoords.cell import Cell, CellList
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
        storm_data = load_thunderstorm(os.path.join(f_path, 'test_data/ds3/storm_table.csv'))
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
        self.data = load_testdata('ds1')
        f_path = os.path.dirname(os.path.realpath(__file__))
        self.storm_cells_1 = load(os.path.join(f_path, 'test_data/test_single_spot_storm.hdf5'))
        self.storm_cells_2 = load(os.path.join(f_path, 'test_data/test_double_spot_storm.hdf5'))

        cells_no_flu = []
        for c in self.storm_cells_2:
            d = Data()
            d.add_data(c.data.binary_img, 'binary')
            d.add_data(c.data.data_dict['storm_1'], 'storm', 'storm_1')
            d.add_data(c.data.data_dict['storm_2'], 'storm', 'storm_2')
            cell = Cell(d)
            cells_no_flu.append(cell)

        self.storm_cells_2_no_flu = CellList(cells_no_flu)

    def test_copying(self):
        data_copy = self.data.copy()
        for k, v in self.data.data_dict.items():
            self.assertArrayEqual(v, data_copy.data_dict[k])

        i = self.data.data_dict['fluorescence'][5, 10, 10]
        self.data.data_dict['fluorescence'][5, 10, 10] += 20
        self.assertEqual(self.data.data_dict['fluorescence'][5, 10, 10], i + 20)
        self.assertEqual(i, data_copy.data_dict['fluorescence'][5, 10, 10])

    def test_rotation(self):
        data_rotated = self.data[:2].rotate(60)
        rotated = scipy_rotate(self.data.binary_img[:2], -60, mode='nearest', axes=(-1, -2))
        self.assertArrayEqual(rotated, data_rotated.binary_img)
        self.assertEqual(len(data_rotated), 2)

    def test_rotation_storm(self):
        for cell in self.storm_cells_1:
            for th in np.arange(90, 370, 90):
                data_r = cell.data.copy().rotate(th)
                flu = data_r.data_dict['fluorescence']
                storm = data_r.data_dict['storm']
                x, y = storm['x'], storm['y']

                nc = Cell(data_r, init_coords=False)
                nc.coords.shape = data_r.shape
                x_fl = np.sum(nc.coords.x_coords * flu) / np.sum(flu)
                y_fl = np.sum(nc.coords.y_coords * flu) / np.sum(flu)

                self.assertAlmostEqual(x[0], np.array(x_fl), 2)
                self.assertAlmostEqual(y[0], np.array(y_fl), 2)

        # https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python/13849249#13849249
        # for cell in self.storm_cells_2_no_flu:
        #     storm = cell.data.data_dict['storm_1']
        #     x1, y1 = storm['x'][0], storm['y'][0]
        #
        #     storm = cell.data.data_dict['storm_2']
        #     x2, y2 = storm['x'][0], storm['y'][0]
        #
        #     d = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        #     angle = np.arctan2(y1-y2, x1-x2)
        #
        #     data = cell.data.copy()
        #     for th in range(0, 740, 20):
        #         data_r = data.rotate(th)
        #
        #         storm = data_r.data_dict['storm_1']
        #         x1, y1 = storm['x'][0], storm['y'][0]
        #
        #         storm = data_r.data_dict['storm_2']
        #         x2, y2 = storm['x'][0], storm['y'][0]
        #
        #         d1 = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        #         self.assertAlmostEqual(d, d1, 5)
        #
        #         angle1 = np.arctan2(y1-y2, x1-x2)
        #         rounded = np.round((angle - angle1)*(180/np.pi) + th, 10)
        #         self.assertAlmostEqual(rounded % 360, 0)

    def test_iteration(self):
        for i, d in enumerate(self.data):
            with self.subTest(i=i):
                self.assertArrayEqual(self.data.binary_img[i], d.binary_img)

        self.assertEqual(len(self.data), 10)


if __name__ == '__main__':
    unittest.main()
