"""
MAP-Elites archiving strategy.
General implementation of the MAP-Elites archiving strategy used in MAP-Elites algorithm
together with a multiobjective elite criteria variant
Author: Luis Andrés Eguiarte-Morett (Github: @leguiart)
License: MIT.
"""


import glob
import numpy as np
import itertools
import copy
import os
import pickle
import shutil
from typing import List, Callable, TypeVar
from sklearn.neighbors import KDTree

from qd_pymoo.Evaluators.IEvaluator import IEvaluationFunction
from qd_pymoo.Evaluators.NoveltyEvaluator import NoveltyEvaluatorKD

T = TypeVar("T")

# Given a bin number, compute it's index
# ppd : points per dimension
def bin2index(id, ppd):
    dim = ppd.size
    ind = np.zeros((dim), dtype=np.int32) #indice
    for i in range(dim):
        p = np.prod(ppd[0:dim - i - 1])
        ind[dim - i - 1] = id // p
        id = id % p
    return ind

# Given some index, compute it's bin number
# ppd : points per dimension
def index2bin(ind, ppd):
    dim = ppd.size
    bin_id = 0
    for i in range(dim):
        bin_id += ind[dim - i - 1]*np.prod(ppd[0:dim - i - 1]) #producto
    return int(bin_id)

def descriptor2index(x : np.ndarray, l: np.ndarray, h: np.ndarray, bpd : np.ndarray):
    res = np.min(np.vstack([(x - l) // h, bpd - 1]), axis=0)
    return res.astype(int)

def std_descriptor_func(x : np.ndarray) -> np.ndarray:
    return x

class MAP_ElitesArchive(IEvaluationFunction, object):
    def __init__(self, name : str,
                lower_lim : np.ndarray,
                upper_lim : np.ndarray,
                bpd : np.ndarray,
                extract_descriptors_func : Callable[[T], List] = std_descriptor_func) -> None:

        self.name = name
        self.upper_lim = upper_lim
        self.lower_lim = lower_lim
        self.bpd = bpd
        self.extract_descriptors_func = extract_descriptors_func

        self.ppd = bpd + 1
        self.dim = self.ppd.size
        self.delta_bin = (self.upper_lim - self.lower_lim) / self.bpd
        self.index2BinCache = {}
        points_per_feature = []
        indexes_per_feature = []

        for i in range(self.dim):
            points = list(np.linspace(self.lower_lim[i], self.upper_lim[i], self.ppd[i]))
            indexes = list(range(len(points) - 1))
            points_per_feature += [points]
            indexes_per_feature += [indexes]
            
        bin_space = []
        for element in itertools.product(*points_per_feature):
            li = list(element)
            li.reverse()
            bin_space += [li]
        self.bin_space = np.vstack(bin_space)

        for j, indx in enumerate(itertools.product(*indexes_per_feature)):
            li_indx = list(indx)
            li_indx.reverse()
            self.index2BinCache[tuple(li_indx)] = j

        self.filled_indices = {}
        self.archive = [None for _ in range(len(self.index2BinCache))]

    def __getitem__(self, i):
        return self.archive[i]

    def __len__(self) -> int:
        return len(self.archive)

    def evaluation_fn(self, X : List[T], *args, **kwargs) -> List[T]:
        pop_fitness_scores = kwargs['fitness_scores']
        for ind, fitness_score in zip(X, pop_fitness_scores):
            self.try_add(ind, fitness_score)     
        return pop_fitness_scores

    def feature_descriptor_idx(self, x):
        b_x = self.extract_descriptors_func(x)
        index = descriptor2index(b_x, self.lower_lim, self.delta_bin, self.bpd)
        return self.index2BinCache[tuple(index)]

    def try_add(self, x, fitness_score):
        i = self.feature_descriptor_idx(x)
        xe = self[i]
        if xe is None or xe[1] > fitness_score:
            self.filled_indices[i] = fitness_score
            self.archive[i] = (copy.deepcopy(x), fitness_score)
            return True
        else:
            return False
    
    def coverage(self):
        return len(self.filled_indices)/len(self)

    def qd_scores(self, attributes = {'fitness':'qd-score_f'}):
        scores = {}
        for attr in attributes.keys():
            scores[attributes[attr]] = 0

        for i in self.filled_indices:
            for attr in attributes.keys():
                scores[attributes[attr]] += getattr(self.archive[i][0], attr)
        return scores
        
    def update_existing_archive(self, novelty_evaluator : NoveltyEvaluatorKD, novelty_attribute : str, elite_feature : str):

        novelty_hashtable = {}
        union_matrix = []

        # Compute the Union of Novelty archive and the population
        for individual in novelty_evaluator.novelty_archive:
            if individual.md5 not in novelty_hashtable:
                novelty_hashtable[individual.md5] = individual
                union_matrix += [novelty_evaluator.vector_extractor(individual)]
        
        union_matrix = np.array(union_matrix)
        kd_tree = KDTree(union_matrix)

        for indx in self.filled_indices:
            current_elite = self[indx][0]

            if current_elite.md5 in novelty_hashtable:
                updated_elite_novelty = getattr(novelty_hashtable[current_elite.md5], novelty_attribute)
            else:
                distances, _ = kd_tree.query([novelty_evaluator.vector_extractor(current_elite)], 
                                                min(novelty_evaluator.k_neighbors, len(union_matrix)))
                updated_elite_novelty = np.mean(distances)

            setattr(current_elite, novelty_attribute, updated_elite_novelty)
            self.filled_indices[indx] = -getattr(current_elite, elite_feature)
            self.archive[indx] = (current_elite, -getattr(current_elite, elite_feature))

      