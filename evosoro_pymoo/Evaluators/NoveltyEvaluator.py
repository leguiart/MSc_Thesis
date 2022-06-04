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

from evosoro_pymoo.Evaluators.IEvaluator import IEvaluator
from common.Utils import timeit

logger = logging.getLogger(f"__main__.{__name__}")

def std_is_valid(x):
    return True

class NoveltyEvaluator(IEvaluator):
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
    def __init__(self, distance_metric, novelty_name, novelty_threshold = 30., novelty_floor = .25, 
                min_novelty_archive_size = 1, k_neighbors = 10, max_novelty_archive_size = None, 
                max_iter = 100, nslc_neighbors_name = None, is_valid_func = std_is_valid):
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
        self.novelty_threshold = novelty_threshold
        self.novelty_floor = novelty_floor
        self.min_novelty_archive_size = min_novelty_archive_size
        self.k_neighbors = k_neighbors
        self.max_novelty_archive_size = max_novelty_archive_size
        self.max_iter = max_iter
        self.items_added_in_generation = 0
        self.time_out = 0
        self.its = 0
        self.novelty_archive = []
        self.distance_metric = distance_metric
        self.novelty_name = novelty_name
        self.nslc_neighbors_name = nslc_neighbors_name
        self.is_valid_func = is_valid_func
        self.already_evaluated = set()

    @timeit
    def evaluate(self, X):
        """Evaluates the novelty of each object in a list of objects according to the Novelty-Search algorithm
        X : list
            List of objects which contain a fitness metric definition
        """

        logger.debug("Starting novelty evaluation")
        preliminary_archive = []
        X_copy = X.copy()
        for i in range(len(X)):
            if self.is_valid_func(X[i]):
                if not X[i].md5 in self.already_evaluated:
                    novelty, kn_neighborsf = self._average_knn_distance(X[i], X_copy)
                    # Set novelty and in case it is needed (NSLC), the fitness of the kn neighbors
                    setattr(X[i], self.novelty_name, novelty)
                    if self.nslc_neighbors_name is not None:
                        setattr(X[i], self.nslc_neighbors_name, kn_neighborsf)
                    if(getattr(X[i], self.novelty_name) > self.novelty_threshold or len(preliminary_archive) + len(self.novelty_archive) < self.min_novelty_archive_size):
                        self.items_added_in_generation+=1
                        # Should add individuals to the archive just yet? if so, could end up counting twice the same individual
                        self.novelty_archive += [copy.deepcopy(X[i])]
                        # Instead add to a preliminary archive for the current population
                        # preliminary_archive += [copy.deepcopy(X[i])]
                        # self.already_evaluated.add(X[i].md5)

        # Now we can actually add the individuals to the archive
        self.novelty_archive += preliminary_archive
        logger.debug("Finished novelty evaluation")

        self._adjust_archive_settings()
        return X
    
    def _adjust_archive_settings(self):
        # If we've exceeded the maximum size of the novelty archive
        if not self.max_novelty_archive_size is None and len(self.novelty_archive) > self.max_novelty_archive_size:
            # We have to recompute the novelty of all the individuals that were previously on the archive,
            # otherwise we could end up evicting the wrong individuals
            for i in range(len(self.novelty_archive)):
                current_ind = self.novelty_archive[i]
                novelty, kn_neighborsf = self._average_knn_distance(current_ind, self.novelty_archive)
                # Set novelty, and in case it is needed (NSLC), the fitness of the kn neighbors
                setattr(current_ind, self.novelty_name, novelty)
                if self.nslc_neighbors_name is not None:
                    setattr(current_ind, self.nslc_neighbors_name, kn_neighborsf)
                self.novelty_archive[i] = current_ind
                
            self.novelty_archive.sort(key = lambda x : getattr(x, self.novelty_name))

            for _ in range(len(self.novelty_archive) - self.max_novelty_archive_size):
                self.novelty_archive.pop(0)


        if self.items_added_in_generation == 0:
            self.time_out+=1
        else:
            self.time_out = 0
        if self.time_out >= 10:
            self.novelty_threshold *= 0.95
            self.novelty_threshold = max(self.novelty_threshold, self.novelty_floor)
            self.time_out = 0
        if self.items_added_in_generation >= 4:
            self.novelty_threshold *= 1.2
        self.items_added_in_generation = 0
    
    def _average_knn_distance(self, artifact, artifacts):
        distances = []
        for a in artifacts:
            distances += [(a.fitness, self.distance_metric(artifact, a))]
        for novel in self.novelty_archive:
            distances += [(novel.fitness, self.distance_metric(artifact,novel))]
        distances.sort(key = lambda x : x[1])
        if len(distances) < self.k_neighbors:
            kn_neighborsf = list(map(lambda x : x[0], distances))
            dists = list(map(lambda x : x[1], distances))
            average_knn_dist = np.average(dists)
        else:
            kn_neighborsf = list(map(lambda x : x[0], distances[0: self.k_neighbors]))
            dists = list(map(lambda x : x[1], distances[0: self.k_neighbors]))
            average_knn_dist = np.average(dists)
        return average_knn_dist, kn_neighborsf


class NSLCQuality(IEvaluator):
    def evaluate(self, X: list, *args, **kwargs) -> list:
        for x in X:
            if x.phenotype.is_valid():
                k = len(x.unaligned_neighbors) 
                x.nslc_quality = k - sum([1 if x.fitness < n else 0 for n in x.unaligned_neighbors])
            
        return X