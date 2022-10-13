
from inspect import isclass
import numpy as np
import pandas as pd
import os
from scipy.spatial import distance_matrix
from pymoo.core.callback import Callback
from typing import List
from evosoro_pymoo.Algorithms.ME_Survival import MESurvival

from experiments.Constants import *
from evosoro_pymoo.common.IAnalytics import IAnalytics
from common.Utils import getsize, readFromDill, readFromPickle, save_json, saveToDill, saveToPickle, timeit
from evosoro_pymoo.Algorithms.MAP_Elites import MAP_ElitesArchive, MOMAP_ElitesArchive



class QD_Analytics(IAnalytics):

    def init_paths(self, experiment_name, json_base_path, csv_base_path):
        self.json_base_path = json_base_path
        self.csv_base_path = csv_base_path

        self.experiment_name = experiment_name
        self.archives_json_name = self.experiment_name + "_archives.json"
        self.archives_json_path = os.path.join(self.json_base_path, self.archives_json_name)
        self.indicator_csv_name = self.experiment_name + ".csv"
        self.indicators_csv_path = os.path.join(self.csv_base_path, self.indicator_csv_name)
        self.stats_csv_name = self.experiment_name + "_stats.csv"
        self.stats_csv_path = os.path.join(self.csv_base_path, self.stats_csv_name)
        self.total_voxels = IND_SIZE[0]*IND_SIZE[1]*IND_SIZE[2]
        self.init_indicator_mapping()
        
        # We are going to have three MAP-Elites archives, for all we are going to store a 2-vector (Fitness, Unaligned Novelty) in each bin, for analysis purposes
        min_max_gr = [(0, self.total_voxels, self.total_voxels + 1), (0, self.total_voxels, self.total_voxels + 1)]
        lower_bound = np.array([0,0])
        upper_bound = np.array([self.total_voxels, self.total_voxels])
        ppd = np.array([self.total_voxels + 1, self.total_voxels + 1])
        #1.- Elites in terms of fitness
        self.map_elites_archive_f = MAP_ElitesArchive("f_elites", self.json_base_path, lower_bound, upper_bound, ppd, self.extract_morpho, bins_type=int)
        #2.- Elites in terms of aligned novelty
        self.map_elites_archive_an = MAP_ElitesArchive("an_elites", self.json_base_path, lower_bound, upper_bound, ppd, self.extract_morpho, bins_type=int)

        self.checkpoint_path = os.path.join(self.json_base_path, f"analytics_checkpoint.pickle")

    def __init__(self, run, method, experiment_name, json_base_path, csv_base_path):
        super().__init__()

        self.init_paths(experiment_name, json_base_path, csv_base_path)
        self.actual_generation = 1
        self.run = run
        self.method = method
 

        self.indicator_stats_set = [
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
            "inipoint_x",
            "inipoint_y",
            "inipoint_z",
            "endpoint_x",
            "endpoint_y",
            "endpoint_z",
            "trayectory_x",
            "trayectory_y",
            "trayectory_z",
            "morphology_active",
            "morphology_passive",
            "unaligned_novelty_archive_fit",
            "aligned_novelty_archive_fit",
            "unaligned_novelty_archive_novelty",
            "aligned_novelty_archive_novelty",
            "qd-score_ff",
            "qd-score_fun",
            "qd-score_fan",
            "qd-score_anf",
            "qd-score_anun",
            "qd-score_anan",
            "coverage",            
            "control_cppn_nodes",
            "control_cppn_edges",
            "control_cppn_ws",
            "morpho_cppn_nodes",
            "morpho_cppn_edges",
            "morpho_cppn_ws",
            "simplified_gene_div",
            "simplified_gene_ne_div",
            "simplified_gene_nws_div"]

        self.indicator_set = ["id",
            "generation",
            "run",
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
            "inipoint_x",
            "inipoint_y",
            "inipoint_z",
            "endpoint_x",
            "endpoint_y",
            "endpoint_z",
            "trayectory_x",
            "trayectory_y",
            "trayectory_z",
            "morphology_active",
            "morphology_passive",
            "control_cppn_nodes",
            "control_cppn_edges",
            "control_cppn_ws",
            "morpho_cppn_nodes",
            "morpho_cppn_edges",
            "morpho_cppn_ws",
            "simplified_gene_div",
            "simplified_gene_ne_div",
            "simplified_gene_nws_div"]


    def start(self, **kwargs):
        resuming_run = kwargs["resuming_run"]
        isNewExperiment = kwargs["isNewExperiment"]

        self.map_elites_archive_f.start(resuming_run = resuming_run)
        self.map_elites_archive_an.start(resuming_run = resuming_run)

        if isNewExperiment:
            if os.path.exists(self.indicators_csv_path):
                os.remove(self.indicators_csv_path)
            
            if os.path.exists(self.stats_csv_path):
                os.remove(self.stats_csv_path)
        
        if resuming_run:
            if os.path.exists(self.indicators_csv_path):
                # Don't load the whole csv onto memory
                with pd.read_csv(self.indicators_csv_path, chunksize=10) as reader:
                    for df in reader:
                        self.indicator_set = df.columns.values.tolist()
                        break
                
    def backup(self):
        saveToPickle(self.checkpoint_path, self)

    def file_recovery(self, *args, **kwargs):
        analytics_from_bkp = readFromPickle(self.checkpoint_path)
        analytics_from_bkp.init_paths(self.experiment_name, self.json_base_path, self.csv_base_path)
        return analytics_from_bkp

    def init_indicator_mapping(self):
        # if self.indicator_mapping:
        #     del self.indicator_mapping
        self.indicator_mapping = {
            "qd-score_ff" : [],
            "qd-score_fun" : [],
            "qd-score_fan" : [],
            "qd-score_anf" : [],
            "qd-score_anun" : [],
            "qd-score_anan" : [],
            "coverage" : [], 
            "id" : [],
            "generation" : [],
            "run" : [],
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
            "endpoint_z": [],
            "inipoint_x": [],
            "inipoint_y": [],
            "inipoint_z": [],
            "trayectory_x": [],
            "trayectory_y": [],
            "trayectory_z" : [],
            "morphology_active": [],
            "morphology_passive": [],
            "control_cppn_nodes": [],
            "control_cppn_edges": [],
            "control_cppn_ws": [],
            "morpho_cppn_nodes": [],
            "morpho_cppn_edges": [],
            "morpho_cppn_ws":[],
            "simplified_gene":[],
            "simplified_gene_no_edges":[],
            "simplified_gene_no_ws":[],
            "simplified_gene_div":[],
            "simplified_gene_ne_div":[],
            "simplified_gene_nws_div":[]
        }

    def extract_morpho(self, x : object) -> List:
        return [x.active, x.passive]

    def extract_endpoint(self, x):
        return [x.finalX, x.finalY, x.finalZ]

    def extract_initpoint(self, x):
        return [x.initialX, x.initialY, x.initialZ]

    def extract_trayectory(self, x):
        return [x.finalX - x.initialX, x.finalY -  x.initialY, x.finalZ - x.finalY]

    @timeit
    def notify(self, algorithm, **kwargs):
        problem = algorithm.problem
        self.actual_generation = algorithm.n_gen
        pop = kwargs['pop']
        child_pop = kwargs['child_pop']
        
        self.init_indicator_mapping()

        if "unaligned_novelty" in problem.evaluators:
            unaligned_archive_key = "unaligned_novelty"
        elif "unaligned_nslc" in problem.evaluators:
            unaligned_archive_key = "unaligned_nslc"

        # self.map_elites_archive_an.update_existing_batch([ind.X for ind in child_pop], problem.evaluators["aligned_novelty"])

        add_to_me_archive = True
        if issubclass(type(algorithm.survival), MESurvival):
            add_to_me_archive = False
            self.map_elites_archive_f = algorithm.survival.me_archive

        self.map_elites_archive_an.update_existing_archive(problem.evaluators["aligned_novelty"])
        
        self.map_elites_archive_an.update_existing_archive(problem.evaluators[unaligned_archive_key])

        self.map_elites_archive_f.update_existing_archive(problem.evaluators["aligned_novelty"])

        self.map_elites_archive_f.update_existing_archive(problem.evaluators[unaligned_archive_key])

        for ind in child_pop:
            if add_to_me_archive: 
                self.map_elites_archive_f.try_add(ind.X)
            self.map_elites_archive_an.try_add(ind.X, quality_metric="aligned_novelty")

        if not algorithm.is_initialized:
            return

        for individual in problem.evaluators["aligned_novelty"].novelty_archive:
            self.indicator_mapping["aligned_novelty_archive_novelty"] += [individual.aligned_novelty]
            self.indicator_mapping["aligned_novelty_archive_fit"] += [individual.fitness]

  
        
        for individual in problem.evaluators[unaligned_archive_key].novelty_archive:
            self.indicator_mapping["unaligned_novelty_archive_novelty"] += [individual.unaligned_novelty]
            self.indicator_mapping["unaligned_novelty_archive_fit"] += [individual.fitness]


        for ind in pop:
            endpoint = self.extract_endpoint(ind.X)
            inipoint = self.extract_initpoint(ind.X)
            trayectory = self.extract_trayectory(ind.X)
            morphology = self.extract_morpho(ind.X)
            self.indicator_mapping["id"] += [ind.X.id]
            self.indicator_mapping["generation"] += [self.actual_generation]
            self.indicator_mapping["run"] += [self.run]
            self.indicator_mapping["md5"] += [ind.X.md5]
            self.indicator_mapping["endpoint"] += [endpoint]
            self.indicator_mapping["inipoint"] += [inipoint]
            self.indicator_mapping["trayectory"] += [trayectory]
            self.indicator_mapping["morphology"] += [morphology]
            self.indicator_mapping["endpoint_x"] += [endpoint[0]]
            self.indicator_mapping["endpoint_y"] += [endpoint[1]]
            self.indicator_mapping["endpoint_z"] += [endpoint[2]]
            self.indicator_mapping["inipoint_x"] += [inipoint[0]]
            self.indicator_mapping["inipoint_y"] += [inipoint[1]]
            self.indicator_mapping["inipoint_z"] += [inipoint[2]]
            self.indicator_mapping["trayectory_x"] += [trayectory[0]]
            self.indicator_mapping["trayectory_y"] += [trayectory[1]]
            self.indicator_mapping["trayectory_z"] += [trayectory[2]]
            self.indicator_mapping["morphology_active"] += [morphology[0]]
            self.indicator_mapping["morphology_passive"] += [morphology[1]]
            self.indicator_mapping["fitness"] += [ind.X.fitness]
            self.indicator_mapping["unaligned_novelty"] += [ind.X.unaligned_novelty]
            self.indicator_mapping["aligned_novelty"] += [ind.X.aligned_novelty]
            self.indicator_mapping["gene_diversity"] += [ind.X.gene_diversity]
            self.indicator_mapping["control_gene_div"] += [ind.X.control_gene_div]
            self.indicator_mapping["morpho_gene_div"] += [ind.X.morpho_gene_div]
            control_cppn = ind.X.genotype.networks[0].graph
            morpho_cppn = ind.X.genotype.networks[1].graph
            self.indicator_mapping["control_cppn_nodes"] += [control_cppn.number_of_nodes()]
            self.indicator_mapping["control_cppn_edges"] += [control_cppn.number_of_edges()]
            self.indicator_mapping["control_cppn_ws"] += [sum([tup[2] for tup in list(control_cppn.edges.data('weight'))])]
            self.indicator_mapping["morpho_cppn_nodes"] += [morpho_cppn.number_of_nodes()]
            self.indicator_mapping["morpho_cppn_edges"] += [morpho_cppn.number_of_edges()]
            self.indicator_mapping["morpho_cppn_ws"] += [sum([tup[2] for tup in list(morpho_cppn.edges.data('weight'))])]
            self.indicator_mapping["simplified_gene"] += [[self.indicator_mapping["control_cppn_nodes"][-1],
                                                            self.indicator_mapping["control_cppn_edges"][-1],
                                                            self.indicator_mapping["control_cppn_ws"][-1],
                                                            self.indicator_mapping["morpho_cppn_nodes"][-1],
                                                            self.indicator_mapping["morpho_cppn_edges"][-1],
                                                            self.indicator_mapping["morpho_cppn_ws"][-1]]]
            self.indicator_mapping["simplified_gene_no_edges"] += [[self.indicator_mapping["control_cppn_nodes"][-1],
                                                self.indicator_mapping["control_cppn_ws"][-1],
                                                self.indicator_mapping["morpho_cppn_nodes"][-1],
                                                self.indicator_mapping["morpho_cppn_ws"][-1]]]
            self.indicator_mapping["simplified_gene_no_ws"] += [[self.indicator_mapping["control_cppn_nodes"][-1],
                                                self.indicator_mapping["control_cppn_edges"][-1],
                                                self.indicator_mapping["morpho_cppn_nodes"][-1],
                                                self.indicator_mapping["morpho_cppn_edges"][-1]]]
            

        self.indicator_mapping["morpho_div"]= list(np.mean(distance_matrix(self.indicator_mapping["morphology"], self.indicator_mapping["morphology"]), axis=1))
        self.indicator_mapping["endpoint_div"] = list(np.mean(distance_matrix(self.indicator_mapping["endpoint"], self.indicator_mapping["endpoint"]), axis=1))
        self.indicator_mapping["trayectory_div"] = list(np.mean(distance_matrix(self.indicator_mapping["trayectory"], self.indicator_mapping["trayectory"]), axis=1))
        self.indicator_mapping["simplified_gene_div"] = list(np.mean(distance_matrix(self.indicator_mapping["simplified_gene"], self.indicator_mapping["simplified_gene"]), axis=1))
        self.indicator_mapping["simplified_gene_ne_div"] = list(np.mean(distance_matrix(self.indicator_mapping["simplified_gene_no_edges"], self.indicator_mapping["simplified_gene_no_edges"]), axis=1))
        self.indicator_mapping["simplified_gene_nws_div"] = list(np.mean(distance_matrix(self.indicator_mapping["simplified_gene_no_ws"], self.indicator_mapping["simplified_gene_no_ws"]), axis=1))
        self.indicator_mapping["coverage"] += [self.map_elites_archive_f.coverage()]
        f_qd_scores = self.map_elites_archive_f.qd_scores({'fitness':'qd-score_ff', 'unaligned_novelty':'qd-score_fun', 'aligned_novelty':'qd-score_fan'})
        an_qd_scores = self.map_elites_archive_an.qd_scores({'fitness':'qd-score_anf', 'unaligned_novelty':'qd-score_anun', 'aligned_novelty':'qd-score_anan'})

        for score in f_qd_scores.keys():
            self.indicator_mapping[score] += [f_qd_scores[score]]

        for score in an_qd_scores.keys():
            self.indicator_mapping[score] += [an_qd_scores[score]]

        indicator_df = self.indicator_df()
        indicator_df.to_csv(self.indicators_csv_path, mode='a', header=not os.path.exists(self.indicators_csv_path), index = False)

        stats_df = self.indicator_stats_df()
        stats_df.to_csv(self.stats_csv_path, mode='a', header=not os.path.exists(self.stats_csv_path), index = False)

        
    def save_archives(self, algorithm):

        archives = {
            "map_elites_archive_f" : [],
            "map_elites_archive_an" : [],
            "novelty_archive_f" : [],
            "novelty_archive_an" : []
        }
        if issubclass(type(algorithm.survival), MESurvival):
            add_to_me_archive = False
            self.map_elites_archive_f = algorithm.survival.me_archive
        # Coverage is the same for all archives
        for i in range(len(self.map_elites_archive_f)):
            xf = self.map_elites_archive_f[i]
            xan = self.map_elites_archive_an[i]
            # If one is None, all are None
            if xf is not None:
                archives["map_elites_archive_f"] += [[xf.md5, xf.id, xf.fitness, xf.unaligned_novelty, xf.aligned_novelty]]
                archives["map_elites_archive_an"] += [[xan.md5, xan.id, xan.fitness, xan.unaligned_novelty, xan.aligned_novelty]]
                # saveToPickle(f"{self.map_elites_archive_f.archive_path}/elite_{i}.pickle", xf)
                # saveToPickle(f"{self.map_elites_archive_an.archive_path}/elite_{i}.pickle", xan)
            else:
                archives["map_elites_archive_f"] += [0]
                archives["map_elites_archive_an"] += [0]
        save_json(self.archives_json_path, archives)


    def indicator_df(self):
        return pd.DataFrame({k : self.indicator_mapping[k] for k in self.indicator_set})
            

    def indicator_stats_df(self):
        d = {"Indicator":[], "Best":[], "Worst":[], "Average":[], "STD":[], "Median":[], "Generation":[], "Run":[], "Method":[]}

        for key in self.indicator_stats_set:
            # Population fitness
            # d["Indicator"] += [key.replace("_", " ")]
            d["Indicator"] += [key]
            arr = np.array(self.indicator_mapping[key])
            d["Best"] += [np.nanmax(arr)]
            d["Worst"] += [np.nanmin(arr)]
            d["Average"] += [np.nanmean(arr)]
            d["STD"] += [np.nanstd(arr)]
            d["Median"] += [np.nanmedian(arr)]
            d["Generation"] += [self.actual_generation]
            d["Run"] += [self.run]
            d["Method"] += [self.method]

        return pd.DataFrame(d)
        

