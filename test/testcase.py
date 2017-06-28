import unittest
import numpy as np

class ArrayTestCase(unittest.TestCase):
    def assertArrayEqual(self, arr1, arr2):
        print np.testing.assert_array_equal(arr1, arr2)
        np.testing.assert_array_equal(arr1, arr2, verbose=True)