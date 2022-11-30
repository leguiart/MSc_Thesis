
import unittest
import random
import numpy as np
import logging

from qd_pymoo.Evaluators.NoveltyEvaluator import NoveltyEvaluatorKD

# Initializing the random number generator for reproducibility
random_seed = 10

# create logger with __name__
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# create file handler which logs only warning level messages
fh = logging.FileHandler('ns_tests.log')
fh.setLevel(logging.WARNING)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)

def std_vector_extractor(x):
    return x

class TestNS(unittest.TestCase):
    def setUp(self) -> None:
        self.novelty_evaluator = NoveltyEvaluatorKD('TestNSEvaluator', k_neighbors=3, novelty_threshold=5., min_novelty_archive_size=5)

    def assertNoveltyScores(self, actual_novelty_scores, expected_novelty_scores):
        for i, expected_novelty_score in enumerate(expected_novelty_scores):
            actual_novelty_score = actual_novelty_scores[i]
            self.assertAlmostEqual(actual_novelty_score, expected_novelty_score, places=3)   

    def assertNoveltyArchive(self, actual_novelty_archive, expected_novelty_archive, vector_func = std_vector_extractor):
        for i, expected_archive_element in enumerate(expected_novelty_archive):
            actual_archive_element = vector_func(actual_novelty_archive[i])
            self.assertListAlmostEqual(list(actual_archive_element), list(expected_archive_element), places=3)

    def assertListAlmostEqual(self, list1, list2, places = 7):
        self.assertEqual(len(list1), len(list2))
        for i, elem1 in enumerate(list1):
            elem2 = list2[i]
            self.assertAlmostEqual(elem1, elem2, places = places)


    def tearDown(self) -> None:
        del self.novelty_evaluator

    def test_matrix_novelty_evaluation(self):
        np.random.seed(random_seed)
        random.seed(random_seed)
        points = 10 * np.random.random_sample((5,2)) - 5

        actual_novelty_scores_vector, actual_indexes_matrix = self.novelty_evaluator._evaluate_novelty(points)
        expected_novelty_scores_vector = np.array([5.6254, 5.7261, 4.1279, 5.7426, 5.4511])
        expected_indexes_matrix = np.array([[1,2,4], [0,2,3], [0,1,4], [1,2,4], [0,2,3]], dtype=np.int32)
        self.assertNoveltyScores(actual_novelty_scores_vector, expected_novelty_scores_vector)
        
        for i, expected_indexes_vector in enumerate(expected_indexes_matrix):
            actual_indexes_vector = actual_indexes_matrix[i]
            self.assertCountEqual(actual_indexes_vector, expected_indexes_vector)

    def test_novelty_evaluation(self):
        np.random.seed(random_seed)
        random.seed(random_seed)
        expected_novelty_scores_list = [np.array([11.25082054, 11.45233782,  8.25582219, 11.48524426, 10.90231331]),
                                        np.array([5.98900197, 7.7556143 , 3.5193219 , 4.74067765, 4.82130199,
                                                  3.9076164 , 4.77278036, 4.3957483 , 3.36937791, 2.7006857 ]),
                                        np.array([6.03755749, 6.51159013, 5.50899449, 2.58614747, 4.6105312 ,
                                                  3.84905613, 1.91039436, 3.18817302, 2.66012999, 4.1908103 ]),
                                        np.array([5.13471626, 3.80550699, 4.38215468, 7.11827582, 3.98874852,
                                                  4.47802531, 5.20162683, 5.31742259, 4.52265995, 3.90849158]),
                                        np.array([4.93011631, 4.31798086, 2.9023319 , 6.42290401, 6.36709623,
                                                  2.74905449, 4.35317016, 2.7433784 , 4.02496854, 2.02007709]),
                                        np.array([3.94155097, 8.50062087, 2.12399261, 8.45274236, 2.09560911,
                                                  4.20563557, 4.27117421, 4.49815865, 4.47888124, 2.06023906]),
                                        np.array([4.60914541, 3.1369434 , 2.4885875 , 2.12086919, 7.04645637,
                                                  2.58546377, 3.37750665, 3.18881442, 2.13240001, 3.15072217]),
                                        np.array([3.5869281 , 2.95209228, 7.17369254, 3.113052  , 8.89602867,
                                                  1.62862685, 2.69140977, 1.83896534, 2.48932477, 5.27220452])]
        expected_novelty_archive_scores_list = [np.array([11.25082054, 11.45233782,  8.25582219, 11.48524426, 10.90231331]),
                                                np.array([5.83530273, 3.89790346, 3.62330036, 6.12062151, 7.82760069]),
                                                np.array([7.58538774, 4.51979398, 5.86835557, 2.38837272, 5.04789701]),
                                                np.array([4.78201728, 4.32810338, 5.52234447, 5.14633486, 5.04579115, 5.20162683, 5.31742259]),
                                                np.array([6.99250477, 4.14981923, 3.59564774, 6.09616391, 3.55231701, 2.16885809, 4.3910944]),
                                                np.array([6.19240569, 3.04828742, 3.94974581, 5.16380973, 6.82086433, 3.931529  , 4.88414698]),
                                                np.array([4.26333728, 5.45802069, 5.04465392, 2.55324399, 3.22361455, 3.33123646, 2.78469438]),
                                                np.array([6.71231623, 1.90158963, 5.72765171, 5.62620976, 6.25314958, 2.53902357, 3.99338301, 5.27220452])]
        points_list = [20 * np.random.random_sample((5,2)) - 10]
        selected_points_list = [[], [], [6,7], [], [], [], [9]]
        expected_novelty_archive_list = []
        expected_novelty_archive_list += [points_list[0]]
        for selected_points in selected_points_list:
            current_points = 20 * np.random.random_sample((10,2)) - 10
            points_list += [current_points]
            if len(selected_points) > 0:
                expected_novelty_archive_list += [np.vstack((expected_novelty_archive_list[-1], current_points[np.array(selected_points), :]))]
            else:
                expected_novelty_archive_list += [expected_novelty_archive_list[-1]]


        for i, points in enumerate(points_list):

            actual_novelty_scores = self.novelty_evaluator.evaluation_fn(points, pop_size = 5)
            expected_novelty_scores = expected_novelty_scores_list[i]
            expected_novelty_archive_scores = expected_novelty_archive_scores_list[i]
            expected_novelty_archive = expected_novelty_archive_list[i]

            self.assertNoveltyScores(actual_novelty_scores, expected_novelty_scores)
            self.assertEqual(len(self.novelty_evaluator.novelty_archive), len(expected_novelty_archive))
            self.assertNoveltyArchive(self.novelty_evaluator.novelty_archive, expected_novelty_archive)
            self.assertNoveltyScores(self.novelty_evaluator.novelty_scores, expected_novelty_archive_scores)

    def test_novelty_evaluation_threshold(self):
        np.random.seed(random_seed)
        random.seed(random_seed)
        expected_novelty_scores_list = [np.array([5.6254, 5.7261, 4.1279, 5.7426, 5.4511]),
                                        np.array([3.0274, 4.4736, 2.3491, 2.8066, 2.5531]),
                                        np.array([2.2045, 2.3863, 2.1978, 1.7428, 1.6989])]
        expected_novelty_archive_scores_list = [np.array([5.6254, 5.7261, 4.1279, 5.7426, 5.4511]),
                                        np.array([4.0316, 2.4071, 3.1010, 4.2500, 4.6679]),
                                        np.array([3.7181, 2.1857, 2.0747, 3.4621, 3.9138])]
        expected_novelty_archive = 10 * np.random.random_sample((5,2)) - 5
        for i, expected_novelty_scores in enumerate(expected_novelty_scores_list):
            if i > 0:
                points = 10 * np.random.random_sample((5,2)) - 5
            else:
                points = expected_novelty_archive
            actual_novelty_scores = self.novelty_evaluator.evaluation_fn(points, pop_size = 5)
            self.assertNoveltyScores(actual_novelty_scores, expected_novelty_scores)
            self.assertEqual(len(self.novelty_evaluator.novelty_archive), 5)
            self.assertNoveltyArchive(self.novelty_evaluator.novelty_archive, expected_novelty_archive)
            self.assertNoveltyScores(self.novelty_evaluator.novelty_scores, expected_novelty_archive_scores_list[i])



if __name__ == '__main__':
    unittest.main()