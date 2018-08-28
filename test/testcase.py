import unittest
import numpy as np

class ArrayTestCase(unittest.TestCase):
    def assertArrayEqual(self, arr1, arr2, verbose=True):
        np.testing.assert_array_equal(arr1, arr2, verbose=verbose)

    def assertArrayAlmostEqual(self, arr1, arr2, decimal=6, verbose=True):
        np.testing.assert_array_almost_equal(arr1, arr2, decimal=decimal, verbose=verbose)