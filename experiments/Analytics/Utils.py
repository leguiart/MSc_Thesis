import random
import json
import numpy as np
import os
import pandas as pd
import sys
from pymoo.core.callback import Callback
from pymoo.factory import get_performance_indicator

from .Constants import *
sys.path.append(os.getcwd() + "/..")
from Algorithms.MAP_Elites import MAP_ElitesArchive, MOMAP_ElitesArchive


def setRandomSeed(seed):
    random.seed(seed)  # Initializing the random number generator for reproducibility
    np.random.seed(seed)

def get_extreme_points_c(F, ideal_point, extreme_points=None):
    # calculate the asf which is used for the extreme point decomposition
    weights = np.eye(F.shape[1])
    weights[weights == 0] = 1e6

    # add the old extreme points to never loose them for normalization
    _F = F
    if extreme_points is not None:
        _F = np.concatenate([extreme_points, _F], axis=0)

    # use __F because we substitute small values to be 0
    __F = _F - ideal_point
    __F[__F < 1e-3] = 0

    # update the extreme points for the normalization having the highest asf value each
    F_asf = np.max(__F * weights[:, None, :], axis=2)

    I = np.argmin(F_asf, axis=1)
    extreme_points = _F[I, :]

    return extreme_points

def readFromJson(filename):
    if os.path.exists(filename):            
        with open(filename, 'r') as fp:           
            return json.load(fp)
    else:
        return {}

def writeToJson(filename, content):
    with open(filename, 'w') as fp:           
        json.dump(content, fp)

def save_json(filename, content):
    if os.path.exists(filename): 
        with open(filename, 'a') as fp:           
            fp.write('\n')
            json.dump(content, fp)
    else:
        with open(filename, 'w') as fp:           
            json.dump(content, fp)

def countFileLines(filename):
    if os.path.exists(filename): 
        with open(filename, 'r') as fh:
            count = 0
            for _ in fh:
                count += 1
        return count
    else:
        return 0

def readFirstJson(filename):
    if os.path.exists(filename): 
        with open(filename, 'r') as fh:
            line = fh.readline()
            j = json.loads(line)
            return j
    else:
        return {}


class MOEA_Analytics(Callback):
    def __init__(self, n_dim):
        super().__init__()
        self.front_history = {"fronts" : [], "hyper_volumes" : []}
        self.ideal_point = np.full(n_dim, np.inf)
        self.worst_point = np.full(n_dim, -np.inf)
        self.extreme_points = None
    
    def notify(self, algorithm):
        first_front = np.array([point.F for point in algorithm.opt])
        self.ideal_point = np.min(np.vstack((self.ideal_point, first_front)), axis=0)
        self.worst_point = np.max(np.vstack((self.worst_point, first_front)), axis=0)
        epsilon = np.zeros(shape=self.worst_point.shape)
        epsilon.fill(0.1)
        ref_point = self.worst_point + epsilon
        self.front_history["fronts"].append(first_front)
        hv = get_performance_indicator("hv", ref_point=ref_point)
        self.extreme_points = get_extreme_points_c(first_front, self.ideal_point, self.extreme_points)
        hypervolume = hv.do(np.vstack((self.extreme_points, self.ideal_point)))
        self.front_history["hyper_volumes"].append(hypervolume)

class QD_Analytics(Callback):
    def __init__(self, run, method):
        super().__init__()
        self.qd_history = {
            "qd-score-f" : [],
            "nd-score-f" : [],
            "qd-score-n" : [],
            "nd-score-n" : [],
            "qd-score-fn" : [],
            "nd-score-fn" :[],
            "coverage" : [], 
            "fitness" : [], 
            "f_archive_progression" : [],
            "n_archive_progression" : [],
            "fn_archive_progression" : [],
            "unaligned_novelty" : []
        }
        self.run = run
        self.method = method
        self.total_voxels = IND_SIZE[0]*IND_SIZE[1]*IND_SIZE[2]
        #We are going to have three MAP-Elites archives, for all we are going to store a 2-vector (Fitness, Unaligned Novelty) in each bin, for analysis purposes
        min_max_gr = [(0, self.total_voxels, self.total_voxels), (0, self.total_voxels, self.total_voxels)]
        #1.- Elites in terms of fitness
        self.map_elites_archive_f = MAP_ElitesArchive(min_max_gr, self.extract_descriptors)
        #2.- Elites in terms of novelty
        self.map_elites_archive_n = MAP_ElitesArchive(min_max_gr, self.extract_descriptors)
        #3.- Elites in terms of both novelty and fitness (Pareto-dominance)
        self.map_elites_archive_fn = MOMAP_ElitesArchive(min_max_gr, self.extract_descriptors)

    def extract_descriptors(self, x):
        return [x.passive, x.active]
    
    def notify(self, pop):
        fitness_li = []
        unaligned_novelty_li = []
        for ind in pop:
            fitness_li += [ind.X.fitness]
            unaligned_novelty_li += [ind.X.unaligned_novelty]
            self.map_elites_archive_f.try_add(ind.X)
            self.map_elites_archive_n.try_add(ind.X, quality_metric="unaligned_novelty")
            self.map_elites_archive_fn.try_add(ind.X)
        
        #We store the flattened versions of each archive to reconstruct them in a posterior data analysis step
        vector_archive_f = []
        vector_archive_n = []
        vector_archive_fn = []
        #We compute qd and nd scores for each archive
        qd_score_f = 0
        nd_score_f = 0
        qd_score_n = 0
        nd_score_n = 0
        qd_score_fn = 0
        nd_score_fn = 0
        #Coverage is the same for all archives
        coverage = 0
        for i in range(len(self.map_elites_archive_f.elites_archive)):
            xf = self.map_elites_archive_f.elites_archive[i]
            xn = self.map_elites_archive_n.elites_archive[i]
            xfn = self.map_elites_archive_fn.elites_archive[i]
            #If one is None, all are None
            if xf is not None:
                coverage += 1
                qd_score_f += xf.fitness
                nd_score_f += xf.unaligned_novelty
                qd_score_n += xn.fitness
                nd_score_n += xn.unaligned_novelty
                qd_score_fn += xfn.fitness
                nd_score_fn += xfn.unaligned_novelty
                vector_archive_f += [[xf.fitness, xf.unaligned_novelty]]
                vector_archive_n += [[xn.fitness, xn.unaligned_novelty]]
                vector_archive_fn += [[xfn.fitness, xfn.unaligned_novelty]]
            else:
                vector_archive_f += [0]
                vector_archive_n += [0]
                vector_archive_fn += [0]
        coverage /= len(vector_archive_f)
        self.qd_history["qd-score-f"] += [qd_score_f]
        self.qd_history["nd-score-f"] += [nd_score_f]
        self.qd_history["qd-score-n"] += [qd_score_n]
        self.qd_history["nd-score-n"] += [nd_score_n]
        self.qd_history["qd-score-fn"] += [qd_score_fn]
        self.qd_history["nd-score-fn"] += [nd_score_fn]
        self.qd_history["coverage"] += [coverage]
        self.qd_history["f_archive_progression"] += [vector_archive_f]
        self.qd_history["n_archive_progression"] += [vector_archive_n]
        self.qd_history["fn_archive_progression"] += [vector_archive_n]
        self.qd_history["fitness"] += [fitness_li]
        self.qd_history["unaligned_novelty"] += [unaligned_novelty_li]

    def to_dataframe(self):
        d = {"Indicator":[], "Best":[], "Average":[], "STD":[], "Generation":[], "Run":[], "Method":[]}
        for gen in range(len(self.qd_history["fitness"])):
            #Computing fitness indicator
            d["Indicator"] += ["Fitness"]
            gen_fitness = np.array(self.qd_history["fitness"][gen])
            d["Best"] += [np.max(gen_fitness)]
            d["Average"] += [np.mean(gen_fitness)]
            d["STD"] += [np.std(gen_fitness)]
            d["Generation"] += [gen + 1]
            d["Run"] += [self.run]
            d["Method"] += [self.method]
            #Computing novelty indicator
            d["Indicator"] += ["Unaligned Novelty"]
            gen_novelty = np.array(self.qd_history["unaligned_novelty"][gen])
            d["Best"] += [np.max(gen_novelty)]
            d["Average"] += [np.mean(gen_novelty)]
            d["STD"] += [np.std(gen_novelty)]
            d["Generation"] += [gen + 1]
            d["Run"] += [self.run]
            d["Method"] += [self.method]
            #Computing coverage
            d["Indicator"] += ["Coverage"]
            d["Best"] += [self.qd_history["coverage"][gen]]
            d["Average"] += [self.qd_history["coverage"][gen]]
            d["STD"] += [0]
            d["Generation"] += [gen + 1]
            d["Run"] += [self.run]
            d["Method"] += [self.method]
            #Computing QD-F            
            d["Indicator"] += ["QD-F"]
            d["Best"] += [self.qd_history["qd-score-f"][gen]]
            d["Average"] += [self.qd_history["qd-score-f"][gen]]
            d["STD"] += [0]
            d["Generation"] += [gen + 1]
            d["Run"] += [self.run]
            d["Method"] += [self.method]
            #Computing ND-F
            d["Indicator"] += ["ND-F"]
            d["Best"] += [self.qd_history["nd-score-f"][gen]]
            d["Average"] += [self.qd_history["nd-score-f"][gen]]
            d["STD"] += [0]
            d["Generation"] += [gen + 1]
            d["Run"] += [self.run]
            d["Method"] += [self.method]
            #Computing QD-N
            d["Indicator"] += ["QD-N"]
            d["Best"] += [self.qd_history["qd-score-n"][gen]]
            d["Average"] += [self.qd_history["qd-score-n"][gen]]
            d["STD"] += [0]
            d["Generation"] += [gen + 1]
            d["Run"] += [self.run]
            d["Method"] += [self.method]            
            #Computing ND-N
            d["Indicator"] += ["ND-N"]
            d["Best"] += [self.qd_history["nd-score-n"][gen]]
            d["Average"] += [self.qd_history["nd-score-n"][gen]]
            d["STD"] += [0]
            d["Generation"] += [gen + 1]
            d["Run"] += [self.run]
            d["Method"] += [self.method]
            #Computing QD-FN
            d["Indicator"] += ["QD-FN"]
            d["Best"] += [self.qd_history["qd-score-fn"][gen]]
            d["Average"] += [self.qd_history["qd-score-fn"][gen]]
            d["STD"] += [0]
            d["Generation"] += [gen + 1]
            d["Run"] += [self.run]
            d["Method"] += [self.method]
            #Computing ND-FN
            d["Indicator"] += ["ND-FN"]
            d["Best"] += [self.qd_history["nd-score-fn"][gen]]
            d["Average"] += [self.qd_history["nd-score-fn"][gen]]
            d["STD"] += [0]
            d["Generation"] += [gen + 1]
            d["Run"] += [self.run]
            d["Method"] += [self.method]
        return pd.DataFrame(d)





def maxFromList(l):
    max_elem = float('-inf')
    for elem in l:
        if elem > max_elem:
            max_elem = elem
    return max_elem



