"""
Novelty search.
General implementation of the Novelty-Search algorithm used to give a novelty score 
as a fitness score in order to drive the selection pressure on an evolutionary algorithm.
Based on the Novelty-Search algorithm found in here: https://github.com/PacktPublishing/Hands-on-Neuroevolution-with-Python/tree/master/Chapter6
Author: Luis Andrés Eguiarte-Morett (Github: @leguiart)
License: MIT.
"""

import numpy as np
import copy
import logging
from typing import Callable, List, TypeVar
from sklearn.neighbors import KDTree

from qd_pymoo.Evaluators.IEvaluator import IEvaluationFunction



logger = logging.getLogger(f"__main__.{__name__}")
T = TypeVar("T")

def std_vector_extractor(x : np.ndarray) -> np.ndarray:
    return x


class NoveltyEvaluatorKD(IEvaluationFunction):
    """
    Novelty-search based phenotype evaluator.
    ...

    Attributes
    ----------
    distance_metric : function
        Function which defines a way to measure a distance
    novelty_threshold : float
        Novelty score dynamic threshold for entrance to novelty archive
    novelty_floor : float
        Lower bound of the novelty threshold
    min_novelty_archive_size : int
        Novelty archive must have at least this number of individuals
    k_neighbors : tuple (float, float)
        K nearest neighbors to compute average distance to in order to get a novelty score
    max_novelty_archive_size : int
        Novelty archive can have at most this number of individuals

    Methods
    -------
    evaluate(artifacts)
        Evaluates the novelty of each artifact in a list of artifacts
    """
    def __init__(self, name : str, k_neighbors = 10, 
                novelty_threshold = 30., novelty_floor = .25, dynamic_threshold = True,
                min_novelty_archive_size = 1, max_novelty_archive_size = None,
                vector_extractor : Callable[[T], List] = std_vector_extractor):
        """
        Parameters
        ----------
        distance_metric : function
            Function which defines a way to measure a distance
        novelty_threshold : float, optional
            Novelty score dynamic threshold for entrance to novelty archive (default is 30)
        novelty_floor : float, optional
            Lower bound of the novelty threshold (default is 0.25)
        min_novelty_archive_size : int, optional (default is 1)
            Novelty archive must have at least this number of individuals
        k_neighbors : int, optional (default is 10)
            K nearest neighbors to compute average distance to in order to get a novelty score
        max_novelty_archive_size : int, optional (default is None)
            Novelty archive can have at most this number of individuals
        """  
        self.name = name      
        self.novelty_threshold = novelty_threshold
        self.novelty_floor = novelty_floor
        self.dynamic_threshold = dynamic_threshold
        self.min_novelty_archive_size = min_novelty_archive_size
        self.k_neighbors = k_neighbors
        self.max_novelty_archive_size = max_novelty_archive_size
        self.items_added_in_generation = 0

        self.time_out = 0
        self.its = 0

        self.novelty_archive = []
        self.novelty_scores = []
        self.vector_extractor = vector_extractor


    def _average_knn_distance(self, kd_matrix, kd_tree : KDTree):
        distances, ind = kd_tree.query(kd_matrix, min(self.k_neighbors + 1, len(kd_matrix)))
        return np.mean(distances[:,1:], axis = 1), ind[:,1:]


    def _evaluate_novelty(self, individuals):
        # Prepare matrix for KD-Tree creation
        kd_matrix = np.array([self.vector_extractor(individuals[i]) 
                            if i < len(individuals) 
                            else self.vector_extractor(self.novelty_archive[i%len(individuals)]) 
                            for i in range(len(individuals) + len(self.novelty_archive))])

        kd_tree = KDTree(kd_matrix)

        return self._average_knn_distance(kd_matrix, kd_tree)

    def _sorted_func(self):

            # self.novelty_archive.sort(key = lambda x : getattr(x, self.novelty_name))
            sorted_novelty_archive = []
            sorted_novelty_scores = []
            for novelty_score, individual in sorted(zip(self.novelty_scores, self.novelty_archive), key= lambda pair : pair[0]):
                sorted_novelty_archive += [individual]
                sorted_novelty_scores += [novelty_score]

            self.novelty_archive = sorted_novelty_archive
            self.novelty_scores = sorted_novelty_scores
            novelty_archive_size = len(self.novelty_archive)

            for _ in range(novelty_archive_size - self.max_novelty_archive_size):
                self.novelty_archive.pop(0)
                self.novelty_scores.pop(0)

    def _adjust_archive_settings(self):
        # If we've exceeded the maximum size of the novelty archive
        if not self.max_novelty_archive_size is None and len(self.novelty_archive) > self.max_novelty_archive_size:
            self._sorted_func()

        if self.dynamic_threshold:
            if self.items_added_in_generation == 0:
                self.time_out+=1
            else:
                self.time_out = 0
            if self.time_out >= 10:
                self.novelty_threshold *= 0.9
                self.novelty_threshold = max(self.novelty_threshold, self.novelty_floor)
                self.time_out = 0
            if self.items_added_in_generation >= 4 and self.its > 0:
                self.novelty_threshold *= 1.2

        self.its+=1
        self.items_added_in_generation = 0


    def evaluation_fn(self, X : List[T], *args, **kwargs) -> List[T]:
        """Evaluates the novelty of each object in a list of objects according to the Novelty-Search algorithm
        X : list
            List of objects which contains a fitness metric definition
        """

        logger.debug("Starting novelty evaluation")

        pop_size = kwargs['pop_size']
        if len(X) == pop_size:
            start_indx = 0
        elif len(X) > pop_size:
            start_indx = len(X) - pop_size
        
        individuals_added = []
        novelty_scores_added = []
        novelty_scores, _ = self._evaluate_novelty(X)
        X_novelty_scores = []
        

        for i in range(len(novelty_scores)):
            novelty = novelty_scores[i]
            
            if i < len(X):
                X_novelty_scores += [novelty]
                if(i >= start_indx and novelty > self.novelty_threshold) or len(self.novelty_archive) < self.min_novelty_archive_size:
                    self.items_added_in_generation+=1
                    ind_copy = copy.deepcopy(X[i])

                    individuals_added += [ind_copy]
                    novelty_scores_added += [novelty]
                        
            else:
                self.novelty_scores[i%len(X)] = novelty

        self.novelty_archive += individuals_added
        self.novelty_scores += novelty_scores_added

        logger.debug(f"{self.items_added_in_generation} were added to {self.name} evaluator")
        self._adjust_archive_settings()
        logger.debug(f"{self.name} evaluator has {self.novelty_threshold} novelty threshold")
        logger.debug(f"{self.name} evaluator currently has {len(self.novelty_archive)} elements")
        logger.debug("Finished novelty evaluation")

        
      
        return -np.array(X_novelty_scores)
    

    def results(self, *args, **kwargs):
        return self.novelty_archive

    def __del__(self):
        del self.novelty_archive
        del self.novelty_scores
        del self.vector_extractor



class NSLCEvaluator(NoveltyEvaluatorKD):

    def __init__(self, name, k_neighbors=10, novelty_threshold=30, 
                novelty_floor=0.25, dynamic_threshold = True,
                min_novelty_archive_size=1, max_novelty_archive_size=None,
                vector_extractor = std_vector_extractor):

        super().__init__(name, k_neighbors, 
                        novelty_threshold, 
                        novelty_floor, 
                        dynamic_threshold,
                        min_novelty_archive_size, 
                        max_novelty_archive_size,
                        vector_extractor)
        self.fitness_scores = []

    def evaluation_fn(self, X : List[T], *args, **kwargs) -> List[T]:
        logger.debug("Starting novelty search with local competition evaluation")

        pop_size = kwargs['pop_size']
        pop_fitness_scores = kwargs['fitness_scores']
        if len(X) == pop_size:
            start_indx = 0
        elif len(X) > pop_size:
            start_indx = len(X) - pop_size

        individuals_added = []
        novelty_scores_added = []
        fitness_scores_added = []
        X_novelty_scores = []
        X_lc_scores = []
        novelty_scores, kn_neighbors_ind = self._evaluate_novelty(X)

        fitness_scores = np.array(list(pop_fitness_scores) + self.fitness_scores)

        for i in range(len(novelty_scores)):
            novelty = novelty_scores[i]
            kn_neighborsf = fitness_scores[kn_neighbors_ind[i]]
            if i < len(X):
                lc_score = sum([1 if pop_fitness_scores[i] > f else 0 for f in kn_neighborsf])

                X_novelty_scores += [novelty]
                X_lc_scores += [lc_score]
                
                if(i >= start_indx and novelty > self.novelty_threshold) or len(self.novelty_archive) < self.min_novelty_archive_size:
                    self.items_added_in_generation+=1
                    ind_copy = copy.deepcopy(X[i])

                    individuals_added += [ind_copy]
                    novelty_scores_added += [novelty]
                    fitness_scores_added += [pop_fitness_scores[i]]
            else:

                self.novelty_scores[i%len(X)] = novelty

        
        self.novelty_archive += individuals_added
        self.novelty_scores += novelty_scores_added
        self.fitness_scores += fitness_scores_added

        logger.debug(f"{self.items_added_in_generation} were added to {self.name} evaluator")
        self._adjust_archive_settings()
        logger.debug(f"{self.name} evaluator has {self.novelty_threshold} novelty threshold")
        logger.debug(f"{self.name} evaluator has currently has {len(self.novelty_archive)} elements")
        logger.debug("Finished novelty search with local competition evaluation")


        return np.array(X_lc_scores), -np.array(X_novelty_scores)

    def _sorted_func(self):

            sorted_novelty_archive = []
            sorted_novelty_scores = []
            sorted_fitness_scores = []
            for novelty_score, individual, fitness_score in sorted(zip(self.novelty_scores, self.novelty_archive, self.fitness_scores), key= lambda pair : pair[0]):
                sorted_novelty_archive += [individual]
                sorted_novelty_scores += [novelty_score]
                sorted_fitness_scores += [fitness_score]

            self.novelty_archive = sorted_novelty_archive
            self.novelty_scores = sorted_novelty_scores
            self.fitness_scores = sorted_fitness_scores
            novelty_archive_size = len(self.novelty_archive)

            for _ in range(novelty_archive_size - self.max_novelty_archive_size):
                self.novelty_archive.pop(0)
                self.novelty_scores.pop(0)
                self.fitness_scores.pop(0)
