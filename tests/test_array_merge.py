import unittest
import numpy as np
from numpy.testing import assert_array_almost_equal
import sys 
import os 

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from fdlbiomass.utils import merge_3d_arrays_with_mean


class TestMerge3DArraysWithMean(unittest.TestCase):

    def test_basic_merge(self):
        arr1 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        arr2 = np.array([[[2, 3], [4, 5]], [[6, 7], [8, 9]]])
        expected = np.array([[[1.5, 2.5], [3.5, 4.5]], [[5.5, 6.5], [7.5, 8.5]]])
        result = merge_3d_arrays_with_mean(arr1, arr2)
        assert_array_almost_equal(result, expected)

    def test_with_nans(self):
        arr1 = np.array([[[1, 2], [3, np.nan]], [[5, 6], [7, 8]]])
        arr2 = np.array([[[2, np.nan], [6, 8]], [[np.nan, 12], [14, 16]]])
        expected = np.array([[[1.5, 2], [4.5, 8]], [[5, 9], [10.5, 12]]])
        result = merge_3d_arrays_with_mean(arr1, arr2)
        assert_array_almost_equal(result, expected)

    def test_all_nans(self):
        arr1 = np.array([[[np.nan, np.nan], [np.nan, np.nan]]])
        arr2 = np.array([[[np.nan, np.nan], [np.nan, np.nan]]])
        expected = np.array([[[np.nan, np.nan], [np.nan, np.nan]]])
        result = merge_3d_arrays_with_mean(arr1, arr2)
        self.assertTrue(np.isnan(result).all())

    def test_different_shapes(self):
        arr1 = np.array([[[1, 2], [3, 4]]])
        arr2 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        with self.assertRaises(ValueError):
            merge_3d_arrays_with_mean(arr1, arr2)

    def test_empty_arrays(self):
        arr1 = np.array([])
        arr2 = np.array([])
        result = merge_3d_arrays_with_mean(arr1, arr2)
        self.assertTrue(np.array_equal(result, np.array([])))

if __name__ == '__main__':
    unittest.main()