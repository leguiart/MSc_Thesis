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
from typing import List, Callable
from sklearn.neighbors import KDTree
from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.sampling.rnd import FloatRandomSampling
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

from common.Utils import readFromJson, readFromPickle, saveToPickle, writeToJson
from evosoro_pymoo.Evaluators.IEvaluator import IEvaluator
from evosoro_pymoo.common.ICheckpoint import ICheckpoint
from evosoro_pymoo.common.IRecoverFromFile import IFileRecovery
from evosoro_pymoo.common.IStart import IStarter

class MAP_Elites(GeneticAlgorithm):
    def __init__(self, 
                batch_size=100, 
                sampling=FloatRandomSampling(), 
                selection=None, 
                mutation=PolynomialMutation, 
                survival=None, 
                **kwargs):
        super().__init__(batch_size, 
                        sampling, 
                        selection, 
                        mutation, 
                        survival, 
                        advance_after_initial_infill = True, 
                        **kwargs)


class MAP_ElitesArchive(ICheckpoint, IEvaluator, IFileRecovery, object):
    def __init__(self, name : str, base_path : str, 
                min_max_gr_li : List[tuple], 
                extract_descriptors_func : Callable[[object], List],
                bins_type = float) -> None:

        bins_per_feat = []
        for min, max, granularity in min_max_gr_li:
            delta_bin_i = (max - min) / granularity
            bins = list(np.linspace(min + delta_bin_i/2, max - delta_bin_i/2, granularity))
            if issubclass(bins_type, int):
                bins = np.rint(bins)
            elif not issubclass(bins_type, float):
                raise ValueError("bins_type parameter must be float or int")
                
            bins_per_feat += [bins]
            
        bin_space = []
        for element in itertools.product(*bins_per_feat):
            bin_space += [list(element)]
        

        self.bin_space = np.vstack(bin_space)
        self.filled_elites_archive = [0 for _ in range(len(self.bin_space))]
        self.extract_descriptors_func = extract_descriptors_func
        self.name = name
        self.base_path = base_path
        self.archive_path = os.path.join(self.base_path, self.name)

        self.coverage = 0 
        self.qd_score = 0 

        self.checkpoint_path = os.path.join(self.base_path, f"{name}_evaluator_checkpoint.pickle")

    def start(self, **kwargs):
        resuming_run = kwargs["resuming_run"]
        if os.path.exists(self.archive_path) and os.path.isdir(self.archive_path):
            if not resuming_run:
                shutil.rmtree(self.archive_path)
                os.mkdir(self.archive_path)
        else:
            os.mkdir(self.archive_path)

    def file_recovery(self):
        return readFromPickle(self.checkpoint_path)

    def backup(self, *args, **kwargs):
        saveToPickle(self.checkpoint_path, self)

    def evaluate(self, X, *args, **kwargs):
        for ind in X:
            self.try_add(ind)
     
        return X

    def _recover_filled_elites(self):
        stored_elites = glob.glob(f"{self.archive_path}/elite_*")
        for elite in stored_elites:
            i = int(elite.split('_')[-1].split('.')[0]) 
            self.filled_elites_archive[i] = 1

    def feature_descriptor(self, x):
        return self.bin_space[self.feature_descriptor_idx(x)]

    def feature_descriptor_idx(self, x):
        b_x = self.extract_descriptors_func(x)
        X = np.tile(b_x, (self.bin_space.shape[0], 1))
        dist_vec = np.sqrt(np.sum((X - self.bin_space)**2, axis = 1))
        return np.argmin(dist_vec)

    def __getitem__(self, i):
        if self.filled_elites_archive[i] != 0:
            return readFromPickle(f"{self.archive_path}/elite_{i}.pickle")
        else:
            return None

    def __len__(self) -> int:
        return len(self.filled_elites_archive)

    def try_add(self, x, quality_metric = "fitness"):
        i = self.feature_descriptor_idx(x)
        xe = self[i]
        if xe is None or getattr(xe, quality_metric) < getattr(x, quality_metric):
            if xe is None:
                self.coverage += 1/len(self)
                self.qd_score += getattr(x, quality_metric)
            else:
                self.qd_score += getattr(x, quality_metric) - getattr(xe, quality_metric)

            self.filled_elites_archive[i] = 1
            saveToPickle(f"{self.archive_path}/elite_{i}.pickle", x)
            return True
        else:
            return False

    def update_existing(self, individual_batch, novelty_evaluator):

        union_hashtable = {}
        union_matrix = []
        archive_matrix = []

        # Compute the Union of Novelty archive and the population
        for individual in individual_batch + novelty_evaluator.novelty_archive:
            if individual.md5 not in union_hashtable:
                union_hashtable[individual.md5] = individual
                union_matrix += [novelty_evaluator.vector_extractor(individual)]
        
        union_matrix = np.array(union_matrix)
        kd_tree = KDTree(union_matrix)
        novelty_metric = novelty_evaluator.novelty_name

        for individual in individual_batch:
            indx = self.feature_descriptor_idx(individual)
            current_elite = self[indx]
            if not current_elite is None:
                old_elite_novelty = getattr(current_elite, novelty_metric)
                self.qd_score -= old_elite_novelty

                if current_elite.md5 in union_hashtable:
                    updated_elite_novelty = getattr(union_hashtable[current_elite.md5], novelty_metric)
                else:
                    distances, _ = kd_tree.query([novelty_evaluator.vector_extractor(current_elite)], min(novelty_evaluator.k_neighbors, len(union_matrix)))
                    updated_elite_novelty = np.mean(distances)
                    # updated_elite_novelty, _ = novelty_evaluator._average_knn_distance([novelty_evaluator.vector_extractor(current_elite)], kd_tree)

                self.qd_score += updated_elite_novelty
                setattr(current_elite, novelty_metric, updated_elite_novelty)
                saveToPickle(f"{self.archive_path}/elite_{indx}.pickle", current_elite)


    
class MOMAP_ElitesArchive(MAP_ElitesArchive):
    def try_add(self, x, metrics = ["fitness", "unaligned_novelty"]):
        i = self.feature_descriptor_idx(x)
        xe = self[i]
        xe_vec = extract_metrics(xe, metrics)
        x_vec = extract_metrics(x, metrics)

        if xe is None or (np.all(xe_vec <= x_vec) and np.any(xe_vec != x_vec)):
            self.filled_elites_archive[i] = 1
            with open(f"{self.archive_path}/elite_{i}.pickle", "wb") as fh:
                pickle.dump(x, fh, protocol=pickle.HIGHEST_PROTOCOL)


def extract_metrics(x, metrics):
    if x is None:
        return None
    return np.array([getattr(x, metric) for metric in metrics])

      