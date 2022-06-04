"""
MAP-Elites archiving strategy.
General implementation of the MAP-Elites archiving strategy used in MAP-Elites algorithm
together with a multiobjective elite criteria variant
Author: Luis Andrés Eguiarte-Morett (Github: @leguiart)
License: MIT.
"""


import numpy as np
import itertools
import copy
import os
import pickle
import shutil
from typing import List, Callable


from evosoro.tools.logging import make_gen_directories, initialize_folders, write_gen_stats
from evosoro.tools.algorithms import Optimizer

class MAP_ElitesOptimizer(Optimizer):
    def __init__(self, sim, env, evaluation_func=...):
        super().__init__(sim, env, evaluation_func=evaluation_func)
    def run(self, *args, **kwargs):
        return super().run(*args, **kwargs)

class MAP_ElitesArchive(object):
    def __init__(self, min_max_gr_li : List[tuple], extract_descriptors_func : Callable[[object], List], name : str) -> None:
        feats = []
        for min, max, granularity in min_max_gr_li:
            feats += [list(np.linspace(min, max, granularity))]
        bc_space = []
        for element in itertools.product(*feats):
            bc_space += [list(element)]
        self.bc_space = np.vstack(bc_space)
        self.filled_elites_archive = [0 for _ in range(len(self.bc_space))]
        self.extract_descriptors_func = extract_descriptors_func
        self.name = name
        if os.path.exists(self.name) and os.path.isdir(self.name):
            shutil.rmtree(self.name)
        os.mkdir(self.name)

    def feature_descriptor(self, x):
        return self.bc_space[self.feature_descriptor_idx(x)]

    def feature_descriptor_idx(self, x):
        b_x = self.extract_descriptors_func(x)
        X = np.tile(b_x, (self.bc_space.shape[0], 1))
        dist_vec = np.sqrt(np.sum((X - self.bc_space)**2, axis = 1))
        return np.argmin(dist_vec)

    def __getitem__(self, i):
        if self.filled_elites_archive[i] != 0:
            with open(f"{self.name}/elite_{i}.pickle", 'rb') as handle:
                x = pickle.load(handle)
            return x
        else:
            return None

    def __len__(self) -> int:
        return len(self.filled_elites_archive)

    def try_add(self, x, quality_metric = "fitness"):
        i = self.feature_descriptor_idx(x)
        xe = self[i]
        if xe is None or getattr(xe, quality_metric) < getattr(x, quality_metric):
            self.filled_elites_archive[i] = 1
            with open(f"{self.name}/elite_{i}.pickle", "wb") as fh:
                pickle.dump(x, fh, protocol=pickle.HIGHEST_PROTOCOL)

    
class MOMAP_ElitesArchive(MAP_ElitesArchive):
    def try_add(self, x, metrics = ["fitness", "unaligned_novelty"]):
        i = self.feature_descriptor_idx(x)
        xe = self[i]
        xe_vec = extract_metrics(xe, metrics)
        x_vec = extract_metrics(x, metrics)

        if xe is None or (np.all(xe_vec <= x_vec) and np.any(xe_vec != x_vec)):
            self.filled_elites_archive[i] = 1
            with open(f"{self.name}/elite_{i}.pickle", "wb") as fh:
                pickle.dump(x, fh, protocol=pickle.HIGHEST_PROTOCOL)


def extract_metrics(x, metrics):
    if x is None:
        return None
    return np.array([getattr(x, metric) for metric in metrics])

      