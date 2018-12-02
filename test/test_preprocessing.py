import tifffile
from colicoords.data_models import Data
from colicoords.fileIO import load_thunderstorm
from colicoords.preprocess import data_to_cells
from test.testcase import ArrayTestCase
import os


class TestDataToCells(ArrayTestCase):

    def setUp(self):
        f_path = os.path.dirname(os.path.realpath(__file__))

        storm = load_thunderstorm(os.path.join(f_path, r'test_data/ds3/storm_table.csv'))
        binary = tifffile.imread(os.path.join(f_path, r'test_data/ds3/binary.tif'))
        fluorescence = tifffile.imread(os.path.join(f_path, r'test_data/ds3/flu.tif'))

        self.storm_ds = Data()
        self.storm_ds.add_data(binary, 'binary')
        self.storm_ds.add_data(fluorescence, 'fluorescence')
        self.storm_ds.add_data(storm, 'storm')

    def test_storm_data_to_cells(self):
        data = self.storm_ds
        cells = data_to_cells(data)
        self.assertEqual(sum([len(c.data.data_dict['storm']) for c in cells]), 40)

        data_copy = data.copy()
        storm_ds = data_copy.data_dict['storm']
        storm_ds['frame'][storm_ds['frame'] == 4] = 5

        cells = data_to_cells(data_copy)
        num_spots = [0, 10, 0, 6, 1, 14, 1, 0, 0, 0, 0, 0, 0]
        for c, spots in zip(cells, num_spots):
            self.assertEqual(len(c.data.data_dict['storm']), spots)
            self.assertIn('storm', c.data.names)
            self.assertIn('storm', c.data.dclasses)
