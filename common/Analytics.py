
import numpy as np
import pandas as pd
import os
from scipy.spatial import distance_matrix
from pymoo.core.callback import Callback
from typing import List

from common.Constants import *
from common.IAnalytics import IAnalytics
from common.Utils import getsize, save_json
from evosoro_pymoo.Algorithms.MAP_Elites import MAP_ElitesArchive, MOMAP_ElitesArchive



class QD_Analytics(IAnalytics):
    def __init__(self, run, method, experiment_name, json_base_path, csv_base_path, resuming_run = False):
        super().__init__()
        self.json_base_path = json_base_path
        self.csv_base_path = csv_base_path
        self.run = run
        self.method = method
        self.experiment_name = experiment_name
        self.archives_json_name = self.experiment_name + "_archives.json"
        self.archives_json_path = os.path.join(self.json_base_path, self.archives_json_name)
        self.indicator_csv_name = self.experiment_name + ".csv"
        self.indicators_csv_path = os.path.join(self.csv_base_path, self.indicator_csv_name)
        self.stats_csv_name = self.experiment_name + "_stats.csv"
        self.stats_csv_path = os.path.join(self.csv_base_path, self.stats_csv_name)
        self.total_voxels = IND_SIZE[0]*IND_SIZE[1]*IND_SIZE[2]
        self.init_indicator_mapping()
        self.actual_generation = 1
        # We are going to have three MAP-Elites archives, for all we are going to store a 2-vector (Fitness, Unaligned Novelty) in each bin, for analysis purposes
        min_max_gr = [(0, self.total_voxels, self.total_voxels), (0, self.total_voxels, self.total_voxels)]
        #1.- Elites in terms of fitness
        self.map_elites_archive_f = MAP_ElitesArchive("f_elites", self.json_base_path, min_max_gr, self.extract_morpho, resuming_run)
        #2.- Elites in terms of aligned novelty
        self.map_elites_archive_an = MAP_ElitesArchive("an_elites", self.json_base_path, min_max_gr, self.extract_morpho, resuming_run)
        #3.- Elites in terms of both aligned novelty and fitness (Pareto-dominance)
        # self.map_elites_archive_anf = MOMAP_ElitesArchive(min_max_gr, self.extract_morpho, "anf_elites")
        self.indicator_stats_set = {
            "qd-score_f",
            "qd-score_an",
            "coverage", 
            "fitness",
            "morphology",
            "unaligned_novelty",
            "aligned_novelty",
            "gene_diversity",
            "control_gene_div",
            "morpho_gene_div",
            "morpho_div",
            "endpoint_div",
            "trayectory_div",
            "endpoint_x",
            "endpoint_y",
            "inipoint_x",
            "inipoint_y",
            "trayectory_x",
            "trayectory_y",
            "morphology_active",
            "morphology_passive",
            "unaligned_novelty_archive_fit",
            "aligned_novelty_archive_fit",
            "unaligned_novelty_archive_novelty",
            "aligned_novelty_archive_novelty",
        }

        self.indicator_set = {
            "id",
            "md5",
            "fitness",
            "unaligned_novelty",
            "aligned_novelty",
            "gene_diversity",
            "control_gene_div",
            "morpho_gene_div",
            "morpho_div",
            "endpoint_div",
            "trayectory_div",
            "endpoint_x",
            "endpoint_y",
            "inipoint_x",
            "inipoint_y",
            "trayectory_x",
            "trayectory_y",
            "morphology_active",
            "morphology_passive"
        }

    def set_generation(self, gen):
        self.actual_generation = gen

    def start(self):
        self.map_elites_archive_f.start()
        self.map_elites_archive_an.start()

    def init_indicator_mapping(self):
        self.indicator_mapping = {
            "qd-score_f" : [0],
            "qd-score_an" : [0],
            "coverage" : [0], 
            "id" : [],
            "md5" : [],
            "fitness" : [],
            "morphology" : [],
            "unaligned_novelty" : [],
            "aligned_novelty" : [],
            "unaligned_novelty_archive_fit" : [],
            "aligned_novelty_archive_fit" : [],
            "unaligned_novelty_archive_novelty" : [],
            "aligned_novelty_archive_novelty" : [],
            "gene_diversity" : [],
            "control_gene_div" : [],
            "morpho_gene_div" :[],
            "endpoint" : [],
            "inipoint" : [],
            "trayectory" : [],
            "morpho_div" : [],
            "endpoint_div" : [],
            "trayectory_div" : [],
            "endpoint_x": [],
            "endpoint_y": [],
            "inipoint_x": [],
            "inipoint_y": [],
            "trayectory_x": [],
            "trayectory_y": [],
            "morphology_active": [],
            "morphology_passive": []
        }



    def extract_morpho(self, x : object) -> List:
        return [x.active, x.passive]

    def extract_endpoint(self, x):
        return [x.finalX, x.finalY]

    def extract_initpoint(self, x):
        return [x.initialX, x.initialY]

    def extract_trayectory(self, x):
        return [x.finalX - x.initialX, x.finalY -  x.initialY]

    
    
    def notify(self, pop, problem):


        for individual in problem.evaluators["aligned_novelty"].novelty_archive:
            self.indicator_mapping["aligned_novelty_archive_novelty"] += [individual.aligned_novelty]
            self.indicator_mapping["aligned_novelty_archive_fit"] += [individual.fitness]

        if "unaligned_novelty" in problem.evaluators:
            unaligned_archive_key = "unaligned_novelty"
        elif "unaligned_nslc" in problem.evaluators:
            unaligned_archive_key = "unaligned_nslc"
        
        for individual in problem.evaluators[unaligned_archive_key].novelty_archive:
            self.indicator_mapping["unaligned_novelty_archive_novelty"] += [individual.unaligned_novelty]
            self.indicator_mapping["unaligned_novelty_archive_fit"] += [individual.fitness]



        self.map_elites_archive_an.update_existing([ind.X for ind in pop], problem.evaluators["aligned_novelty"])

        for ind in pop:
            endpoint = self.extract_endpoint(ind.X)
            inipoint = self.extract_initpoint(ind.X)
            trayectory = self.extract_trayectory(ind.X)
            morphology = self.extract_morpho(ind.X)
            self.indicator_mapping["id"] += [ind.X.id]
            self.indicator_mapping["md5"] += [ind.X.md5]
            self.indicator_mapping["endpoint"] += [endpoint]
            self.indicator_mapping["inipoint"] += [inipoint]
            self.indicator_mapping["trayectory"] += [trayectory]
            self.indicator_mapping["morphology"] += [morphology]
            self.indicator_mapping["endpoint_x"] += [endpoint[0]]
            self.indicator_mapping["endpoint_y"] += [endpoint[1]]
            self.indicator_mapping["inipoint_x"] += [inipoint[0]]
            self.indicator_mapping["inipoint_y"] += [inipoint[1]]
            self.indicator_mapping["trayectory_x"] += [trayectory[0]]
            self.indicator_mapping["trayectory_y"] += [trayectory[1]]
            self.indicator_mapping["morphology_active"] += [morphology[0]]
            self.indicator_mapping["morphology_passive"] += [morphology[1]]
            self.indicator_mapping["fitness"] += [ind.X.fitness]
            self.indicator_mapping["unaligned_novelty"] += [ind.X.unaligned_novelty]
            self.indicator_mapping["aligned_novelty"] += [ind.X.aligned_novelty]
            self.indicator_mapping["gene_diversity"] += [ind.X.gene_diversity]
            self.indicator_mapping["control_gene_div"] += [ind.X.control_gene_div]
            self.indicator_mapping["morpho_gene_div"] += [ind.X.morpho_gene_div]
            self.map_elites_archive_f.try_add(ind.X)
            self.map_elites_archive_an.try_add(ind.X, quality_metric="aligned_novelty")


        self.indicator_mapping["morpho_div"]= list(np.mean(distance_matrix(self.indicator_mapping["morphology"], self.indicator_mapping["morphology"]), axis=1))
        self.indicator_mapping["endpoint_div"] = list(np.mean(distance_matrix(self.indicator_mapping["endpoint"], self.indicator_mapping["endpoint"]), axis=1))
        self.indicator_mapping["trayectory_div"] = list(np.mean(distance_matrix(self.indicator_mapping["trayectory"], self.indicator_mapping["trayectory"]), axis=1))

        self.indicator_mapping["coverage"][0] = self.map_elites_archive_f.coverage
        self.indicator_mapping["qd-score_f"][0] = self.map_elites_archive_f.qd_score
        self.indicator_mapping["qd-score_an"][0] = self.map_elites_archive_an.qd_score

        indicator_df = self.indicator_df()
        indicator_df.to_csv(self.indicators_csv_path, mode='a', header=not os.path.exists(self.indicators_csv_path), index = False)

        stats_df = self.indicator_stats_df()
        stats_df.to_csv(self.stats_csv_path, mode='a', header=not os.path.exists(self.stats_csv_path), index = False)

        self.init_indicator_mapping()
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



    def indicator_df(self):
        return pd.DataFrame({k : self.indicator_mapping[k] for k in self.indicator_set})
            


    def indicator_stats_df(self):
        d = {"Indicator":[], "Best":[], "Average":[], "STD":[], "Generation":[], "Run":[], "Method":[]}

        for key in self.indicator_stats_set:
            # Population fitness
            d["Indicator"] += [key.replace("_", " ")]
            arr = np.array(self.indicator_mapping[key])
            d["Best"] += [np.max(arr)]
            d["Average"] += [np.mean(arr)]
            d["STD"] += [np.std(arr)]
            d["Generation"] += [self.actual_generation]
            d["Run"] += [self.run]
            d["Method"] += [self.method]

        return pd.DataFrame(d)
        

