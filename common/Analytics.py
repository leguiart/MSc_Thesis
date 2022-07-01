
import numpy as np
import pandas as pd
import os
from pymoo.core.callback import Callback
from typing import List

from common.Constants import *
from common.Utils import getsize, save_json
from evosoro_pymoo.Algorithms.MAP_Elites import MAP_ElitesArchive, MOMAP_ElitesArchive



class QD_Analytics(Callback):
    def __init__(self, run, method, experiment_name):
        super().__init__()
        self.run = run
        self.method = method
        self.experiment_name = experiment_name
        self.json_path = self.experiment_name + str(self.run) + ".json" 
        self.archives_json_path = self.experiment_name + "_archives.json"
        self.csv_path = self.experiment_name + str(self.run) + ".csv"
        self.total_voxels = IND_SIZE[0]*IND_SIZE[1]*IND_SIZE[2]
        self.init_qd_history()
        self.actual_generation = 1
        # We are going to have three MAP-Elites archives, for all we are going to store a 2-vector (Fitness, Unaligned Novelty) in each bin, for analysis purposes
        min_max_gr = [(0, self.total_voxels, self.total_voxels), (0, self.total_voxels, self.total_voxels)]
        #1.- Elites in terms of fitness
        self.map_elites_archive_f = MAP_ElitesArchive(min_max_gr, self.extract_morpho, "f_elites")
        #2.- Elites in terms of aligned novelty
        self.map_elites_archive_an = MAP_ElitesArchive(min_max_gr, self.extract_morpho, "an_elites")
        #3.- Elites in terms of both aligned novelty and fitness (Pareto-dominance)
        # self.map_elites_archive_anf = MOMAP_ElitesArchive(min_max_gr, self.extract_morpho, "anf_elites")

    def init_qd_history(self):
        self.qd_history = {
            "qd-score-f" : 0,
            "qd-score-an" : 0,
            "coverage" : 0, 
            "fitness" : [], 
            "unaligned_novelty" : [],
            "aligned_novelty" : [],
            "gene_diversity" : [],
            "control_gene_div" : [],
            "morpho_gene_div" :[]
        }

    def extract_morpho(self, x : object) -> List:
        return [x.passive, x.active]

    def extract_endpoint(self, x):
        return [x.finalX, x.finalY]

    def extract_initpoint(self, x):
        return [x.initialX, x.initialY]

    def extract_trayectory(self, x):
        return [x.finalX - x.initialX, x.finalY -  x.initialY]

    
    
    def notify(self, pop, problem):

        analytics_dict = {
            "endpoints" : [],
            "inipoints" : [],
            "trayectories" : [],
            "aligned_novelties" : []
        }


        self.qd_history["aligned_features"] = [[individual.fitness, individual.aligned_novelty] for individual in problem.evaluators["aligned_novelty"].novelty_archive]
        self.qd_history["unaligned_features"] = [[individual.fitness, individual.unaligned_novelty] for individual in problem.evaluators["unaligned_novelty"].novelty_archive]
        
        self.map_elites_archive_an.update_existing([ind.X for ind in pop], problem.evaluators["aligned_novelty"])

        for ind in pop:
            analytics_dict["endpoints"] += [self.extract_endpoint(ind.X)]
            analytics_dict["inipoints"] += [self.extract_initpoint(ind.X)]
            analytics_dict["trayectories"] += [self.extract_trayectory(ind.X)]

            self.qd_history["fitness"] += [ind.X.fitness]
            self.qd_history["unaligned_novelty"] += [ind.X.unaligned_novelty]
            self.qd_history["aligned_novelty"] += [ind.X.aligned_novelty]
            self.qd_history["gene_diversity"] += [ind.X.gene_diversity]
            self.qd_history["control_gene_div"] += [ind.X.control_gene_div]
            self.qd_history["morpho_gene_div"] += [ind.X.morpho_gene_div]
            self.map_elites_archive_f.try_add(ind.X)
            self.map_elites_archive_an.try_add(ind.X)

        
        self.qd_history["coverage"] = self.map_elites_archive_f.coverage
        self.qd_history["qd-score-f"] = self.map_elites_archive_f.qd_score
        self.qd_history["qd-score-an"] = self.map_elites_archive_an.qd_score


        save_json(self.json_path, analytics_dict)
        df = self.to_dataframe()
        df.to_csv(self.csv_path, mode='a', header=not os.path.exists(self.csv_path), index = False)

        self.init_qd_history()
        self.actual_generation+=1
            
            
    def save_archives(self):

        archives = {
            "map_elites_archive_f" : [],
            "map_elites_archive_an" : []
        }

        # Coverage is the same for all archives
        for i in range(len(self.map_elites_archive_f)):
            xf = self.map_elites_archive_f[i]
            xan = self.map_elites_archive_an[i]
            # If one is None, all are None
            if xf is not None:
                archives["map_elites_archive_f"] += [[xf.fitness, xf.unaligned_novelty, xf.aligned_novelty]]
                archives["map_elites_archive_an"] += [[xan.fitness, xan.unaligned_novelty, xan.aligned_novelty]]
            else:
                archives["map_elites_archive_f"] += [0]
                archives["map_elites_archive_an"] += [0]
        save_json(self.archives_json_path, archives)


    def to_dataframe(self):
        d = {"Indicator":[], "Best":[], "Average":[], "STD":[], "Generation":[], "Run":[], "Method":[]}

        d["Indicator"] += ["Population Fitness"]
        gen_fitness = np.array(self.qd_history["fitness"])
        d["Best"] += [np.max(gen_fitness)]
        d["Average"] += [np.mean(gen_fitness)]
        d["STD"] += [np.std(gen_fitness)]
        d["Generation"] += [self.actual_generation]
        d["Run"] += [self.run]
        d["Method"] += [self.method]
        #Computing unaligned novelty indicator
        d["Indicator"] += ["Unaligned Novelty"]
        gen_novelty = np.array(self.qd_history["unaligned_novelty"])
        d["Best"] += [np.max(gen_novelty)]
        d["Average"] += [np.mean(gen_novelty)]
        d["STD"] += [np.std(gen_novelty)]
        d["Generation"] += [self.actual_generation]
        d["Run"] += [self.run]
        d["Method"] += [self.method]
        #Computing aligned novelty indicator
        d["Indicator"] += ["Aligned Novelty"]
        gen_novelty = np.array(self.qd_history["aligned_novelty"])
        d["Best"] += [np.max(gen_novelty)]
        d["Average"] += [np.mean(gen_novelty)]
        d["STD"] += [np.std(gen_novelty)]
        d["Generation"] += [self.actual_generation]
        d["Run"] += [self.run]
        d["Method"] += [self.method]
        #Computing coverage
        d["Indicator"] += ["Coverage"]
        d["Best"] += [self.qd_history["coverage"]]
        d["Average"] += [self.qd_history["coverage"]]
        d["STD"] += [0]
        d["Generation"] += [self.actual_generation]
        d["Run"] += [self.run]
        d["Method"] += [self.method]
        #Computing QD-F            
        d["Indicator"] += ["QD-F"]
        d["Best"] += [self.qd_history["qd-score-f"]]
        d["Average"] += [self.qd_history["qd-score-f"]]
        d["STD"] += [0]
        d["Generation"] += [self.actual_generation]
        d["Run"] += [self.run]
        d["Method"] += [self.method]
        #Computing QD-AN            
        d["Indicator"] += ["QD-AN"]
        d["Best"] += [self.qd_history["qd-score-an"]]
        d["Average"] += [self.qd_history["qd-score-an"]]
        d["STD"] += [0]
        d["Generation"] += [self.actual_generation]
        d["Run"] += [self.run]
        d["Method"] += [self.method]

        aligned_features = np.array(self.qd_history["aligned_features"])
        unaligned_features = np.array(self.qd_history["unaligned_features"])
        # Computing aligned novelty archive novelty
        d["Indicator"] += ["Aligned Archive Novelty"]
        d["Best"] += [np.max(aligned_features[:,1])]
        d["Average"] += [np.mean(aligned_features[:,1])]
        d["STD"] += [np.std(aligned_features[:,1])]
        d["Generation"] += [self.actual_generation]
        d["Run"] += [self.run]
        d["Method"] += [self.method]
        # Computing aligned novelty archive fitness
        d["Indicator"] += ["Aligned Archive Fitness"]
        d["Best"] += [np.max(aligned_features[:,0])]
        d["Average"] += [np.mean(aligned_features[:,0])]
        d["STD"] += [np.std(aligned_features[:,0])]
        d["Generation"] += [self.actual_generation]
        d["Run"] += [self.run]
        d["Method"] += [self.method]
        # Computing unaligned novelty archive novelty
        d["Indicator"] += ["Unaligned Archive Novelty"]
        d["Best"] += [np.max(unaligned_features[:,1])]
        d["Average"] += [np.mean(unaligned_features[:,1])]
        d["STD"] += [np.std(unaligned_features[:,1])]
        d["Generation"] += [self.actual_generation]
        d["Run"] += [self.run]
        d["Method"] += [self.method]
        # Computing unaligned novelty archive fitness
        d["Indicator"] += ["Unaligned Archive Fitness"]
        d["Best"] += [np.max(unaligned_features[:,0])]
        d["Average"] += [np.mean(unaligned_features[:,0])]
        d["STD"] += [np.std(unaligned_features[:,0])]
        d["Generation"] += [self.actual_generation]
        d["Run"] += [self.run]
        d["Method"] += [self.method]
        
        return pd.DataFrame(d)
