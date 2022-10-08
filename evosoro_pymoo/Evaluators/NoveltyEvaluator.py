"""
Novelty search.
General implementation of the Novelty-Search algorithm used to give a novelty score 
as a fitness score in order to drive the selection pressure on an evolutionary algorithm.
Based on the Novelty-Search algorithm found in here: https://github.com/PacktPublishing/Hands-on-Neuroevolution-with-Python/tree/master/Chapter6
Author: Luis Andrés Eguiarte-Morett (Github: @leguiart)
License: MIT.
"""

import os
import numpy as np
import copy
import logging
from typing import Callable, List, TypeVar
from sklearn.neighbors import KDTree


from evosoro_pymoo.Evaluators.IEvaluator import IEvaluator
from common.Utils import readFromJson, readFromPickle, saveToPickle, timeit, writeToJson
from evosoro_pymoo.common.ICheckpoint import ICheckpoint
from evosoro_pymoo.common.IResults import IResults
from evosoro_pymoo.common.IStart import IStarter


logger = logging.getLogger(f"__main__.{__name__}")

def std_is_valid(x):
    return True

T = TypeVar("T")
class NoveltyEvaluatorKD(IEvaluator[T], IResults, ICheckpoint, IStarter):
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
    def __init__(self, name : str, base_path : str, novelty_name : str, vector_extractor : Callable[[T], List], 
                novelty_threshold = 30., novelty_floor = .25, min_novelty_archive_size = 1, k_neighbors = 10, 
                max_novelty_archive_size = None, max_iter = 100):
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
        self.novelty_name = novelty_name
        self.base_path = base_path
        self.archive_path = os.path.join(self.base_path, self.name)
        self.obj_properties_json_path = os.path.join(self.base_path, f"NS_{self.name}_properties_backup.json")

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
        self.vector_extractor = vector_extractor
        self.individuals_added = []


    def start(self, **kwargs):
        resuming_run = kwargs["resuming_run"]
        self.obj_properties_backup = readFromJson(self.obj_properties_json_path)
        if "novelty_threshold" in self.obj_properties_backup:
            self.novelty_threshold = self.obj_properties_backup["novelty_threshold"]
        if "time_out" in self.obj_properties_backup:
            self.time_out = self.obj_properties_backup["time_out"]
        if "its" in self.obj_properties_backup:
            self.its = self.obj_properties_backup["its"]
        if resuming_run and os.path.exists(self.archive_path) and os.path.isdir(self.archive_path):
            dir_contents = [file for file in os.listdir(self.archive_path) if os.path.isfile(os.path.join(self.archive_path, file))]
            if dir_contents:
                for filename in dir_contents:
                    if filename.endswith(".pickle"):
                        individual = readFromPickle(os.path.join(self.archive_path, filename))
                        self.novelty_archive += [individual]

        else:
            try:
                os.mkdir(self.archive_path)
            except:
                pass


    def backup(self):
        self.obj_properties_backup["its"] = self.its
        self.obj_properties_backup["time_out"] = self.time_out
        self.obj_properties_backup["novelty_threshold"] = self.novelty_threshold
        for ind in self.individuals_added:
            self.pickle_individual(ind)
        writeToJson(self.obj_properties_json_path, self.obj_properties_backup)


    def _evaluate_novelty(self, individuals):
        # Prepare matrix for KD-Tree creation
        kd_matrix = np.array([self.vector_extractor(individuals[i]) 
                            if i < len(individuals) 
                            else self.vector_extractor(self.novelty_archive[i%len(individuals)]) 
                            for i in range(len(individuals) + len(self.novelty_archive))])

        kd_tree = KDTree(kd_matrix)

        return self._average_knn_distance(kd_matrix, kd_tree)


    def pickle_individual(self, individual):
        saveToPickle(f"{self.archive_path}/individual_{individual.id}.pickle", individual)


    def remove_individual_from_backup(self, individual):
        pickle_path = f"{self.archive_path}/individual_{individual.id}.pickle"
        if os.path.exists(pickle_path):
            os.remove(pickle_path)

    def results(self, *args, **kwargs):
        return self.novelty_archive

    @timeit
    def evaluate(self, X : List[T], *args, **kwargs) -> List[T]:
        """Evaluates the novelty of each object in a list of objects according to the Novelty-Search algorithm
        X : list
            List of objects which contain a fitness metric definition
        """

        logger.debug("Starting novelty evaluation")
        
        self.individuals_added = []
        novelty_scores, _ = self._evaluate_novelty(X)
        

        for i in range(len(novelty_scores)):
            novelty = novelty_scores[i]
            
            if i < len(X):
                # Set novelty
                setattr(X[i], self.novelty_name, novelty)   
                if(getattr(X[i], self.novelty_name) > self.novelty_threshold or len(self.novelty_archive) < self.min_novelty_archive_size):
                    self.items_added_in_generation+=1
                    ind_copy = copy.deepcopy(X[i])
                    self.novelty_archive += [ind_copy]
                    self.individuals_added += [ind_copy]
                        
            else:
                setattr(self.novelty_archive[i%len(X)], self.novelty_name, novelty)

        
        logger.debug(f"{self.items_added_in_generation} were added to {self.name} evaluator")
        logger.debug(f"{self.name} evaluator has {self.novelty_threshold} novelty threshold")
        logger.debug(f"{self.name} evaluator currently has {len(self.novelty_archive)} elements")
        logger.debug("Finished novelty evaluation")

        self._adjust_archive_settings()
      
        return X
    

    def _adjust_archive_settings(self):
        # If we've exceeded the maximum size of the novelty archive
        if not self.max_novelty_archive_size is None and len(self.novelty_archive) > self.max_novelty_archive_size:

            self.novelty_archive.sort(key = lambda x : getattr(x, self.novelty_name))

            novelty_archive_size = len(self.novelty_archive)

            for _ in range(novelty_archive_size - self.max_novelty_archive_size):
                removed = self.novelty_archive.pop(0)
                self.remove_individual_from_backup(removed)


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


    def _average_knn_distance(self, kd_matrix, kd_tree : KDTree):
        distances, ind = kd_tree.query(kd_matrix, min(self.k_neighbors + 1, len(kd_matrix)))
        return np.mean(distances[:,1:], axis = 1), ind[:,1:]



class NSLCEvaluator(NoveltyEvaluatorKD):

    def __init__(self, name, base_path, novelty_name, vector_extractor, nslc_quality_name, 
                fitness_name, novelty_threshold=30, novelty_floor=0.25, min_novelty_archive_size=1, 
                k_neighbors=10, max_novelty_archive_size=None, max_iter=100):

        super().__init__(name, base_path, novelty_name, vector_extractor,
                        novelty_threshold, novelty_floor, min_novelty_archive_size, 
                        k_neighbors, max_novelty_archive_size, max_iter)
        self.nslc_quality_name = nslc_quality_name
        self.fitness_name = fitness_name

    def evaluate(self, X : List[T], *args, **kwargs) -> List[T]:
        logger.debug("Starting novelty search with local competition evaluation")

        novelty_scores, kn_neighbors_ind = self._evaluate_novelty(X)

        fitness_scores = np.array([getattr(X[i], self.fitness_name) 
                                if i < len(X) 
                                else getattr(self.novelty_archive[i%len(X)], self.fitness_name)
                                for i in range(len(X) + len(self.novelty_archive))])

        for i in range(len(novelty_scores)):
            novelty = novelty_scores[i]
            kn_neighborsf = fitness_scores[kn_neighbors_ind[i]]
            if i < len(X):

                # Set novelty
                setattr(X[i], self.novelty_name, novelty)
                setattr(X[i], self.nslc_quality_name, self.k_neighbors - sum([1 if getattr(X[i], self.fitness_name) < f else 0 for f in kn_neighborsf]))

                if(getattr(X[i], self.novelty_name) > self.novelty_threshold or len(self.novelty_archive) < self.min_novelty_archive_size):
                    self.items_added_in_generation+=1
                    ind_copy = copy.deepcopy(X[i])
                    self.novelty_archive += [ind_copy]
                    self.individuals_added += [ind_copy]
            else:

                setattr(self.novelty_archive[i%len(X)], self.novelty_name, novelty)
                # setattr(self.novelty_archive[i%len(X)], self.nslc_quality_name, self.k_neighbors - sum([1 if getattr(self.novelty_archive[i%len(X)], self.fitness_name) < f else 0 for f in kn_neighborsf]))

        
        logger.debug(f"{self.items_added_in_generation} were added to {self.name} evaluator")
        logger.debug(f"{self.name} evaluator has {self.novelty_threshold} novelty threshold")
        logger.debug(f"{self.name} evaluator has currently has {len(self.novelty_archive)} elements")
        logger.debug("Finished novelty search with local competition evaluation")

        self._adjust_archive_settings()


        return X
