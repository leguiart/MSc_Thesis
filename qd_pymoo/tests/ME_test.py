
import unittest
import random
import numpy as np

from qd_pymoo.Algorithm.ME_Archive import MAP_ElitesArchive
# Initializing the random number generator for reproducibility
random_seed = 10

class TestME(unittest.TestCase):
    def setUp(self) -> None:
        self.me_archive = MAP_ElitesArchive('TestME_Archive', np.array([0.,0.]), np.array([5., 5.]), np.array([10, 10]))
        self.me_archive2 = MAP_ElitesArchive('TestME_Archive2', np.array([0.,0.]), np.array([125., 125.]), np.array([25, 25]))

    def test_me_creation(self):
        self.assertEqual(len(self.me_archive2), 625)
        self.assertEqual(len(self.me_archive), 100)

    def test_feature_descriptor_bin_id(self):
        self.assertEqual(self.me_archive.feature_descriptor_idx(np.array([2.6,.75])), 15)
        self.assertEqual(self.me_archive.feature_descriptor_idx(np.array([3.,.75])), 16)
        self.assertEqual(self.me_archive.feature_descriptor_idx(np.array([5.,.75])), 19)
        self.assertEqual(self.me_archive.feature_descriptor_idx(np.array([5.,5.])), 99)
        self.assertEqual(self.me_archive2.feature_descriptor_idx(np.array([5,0])), 1)
        self.assertEqual(self.me_archive2.feature_descriptor_idx(np.array([5,1])), 1)
        self.assertEqual(self.me_archive2.feature_descriptor_idx(np.array([5,8])), 26)
        self.assertEqual(self.me_archive2.feature_descriptor_idx(np.array([1,10])), 50)

    def test_add_to_archive(self):
        mocked_pop = np.array([[5,0], [5,1], [5,8], [1, 10], [5, 1]])
        mocked_fitness_scores = np.array([-10., -5., -2., -3., -15])
        self.me_archive2.evaluation_fn(mocked_pop, fitness_scores = mocked_fitness_scores)
        expected = [(np.array([5,1]), -15.), (np.array([5,8]), -2.), (np.array([1,10]), -3.)]
        actual_indexes = [1, 26, 50]
        for i, index in enumerate(actual_indexes):
            actual = self.me_archive2[index]
            self.assertEqual(actual[1], expected[i][1])
            self.assertTrue(np.array_equal(actual[0], expected[i][0]))

        mocked_pop = np.array([[5,0], [10,1], [7,8], [3,32], [120,24]])
        mocked_fitness_scores = np.array([-20., -5., -1., -3., -15])
        self.me_archive2.evaluation_fn(mocked_pop, fitness_scores = mocked_fitness_scores)
        actual_indexes = [1, 2, 26, 50, 150, 124]
        expected = [(np.array([5,0]), -20.), (np.array([10,1]), -5.), (np.array([5,8]), -2.), (np.array([1,10]), -3.), (np.array([3,32]), -3.), (np.array([120,24]), -15.)]
        for i, index in enumerate(actual_indexes):
            actual = self.me_archive2[index]
            self.assertEqual(actual[1], expected[i][1])
            self.assertTrue(np.array_equal(actual[0], expected[i][0]))
        self.assertCountEqual(self.me_archive2.filled_indices, actual_indexes)




    def tearDown(self) -> None:
        del self.me_archive