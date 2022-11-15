
import os
import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
from typing import List

from Constants import *
from evosoro_pymoo.common.IAnalytics import IAnalytics
from evosoro_pymoo.Evaluators.GenotypeDiversityEvaluator import GenotypeDiversityEvaluator
from qd_pymoo.Evaluators.NoveltyEvaluator import NoveltyEvaluatorKD
from qd_pymoo.Algorithm.ME_Archive import MAP_ElitesArchive
from qd_pymoo.Algorithm.ME_Survival import MESurvival
from common.Utils import getsize, readFromDill, readFromPickle, save_json, saveToDill, saveToPickle, timeit
from qd_pymoo.Problems.ME_Problem import BaseMEProblem
from qd_pymoo.Problems.SingleObjectiveProblem import BaseSingleObjectiveProblem


class QD_Analytics(IAnalytics):
    def __init__(self, run, method, experiment_name, json_base_path, csv_base_path,
                un_archive : NoveltyEvaluatorKD, an_archive : NoveltyEvaluatorKD, 
                f_me_archive : MAP_ElitesArchive, an_me_archive : MAP_ElitesArchive, 
                genotypeDivEvaluator : GenotypeDiversityEvaluator):
        super().__init__()
        self.run = run
        self.method = method
        self.init_paths(experiment_name, json_base_path, csv_base_path)
        self.un_archive = un_archive
        self.an_archive = an_archive
        self.f_me_archive = f_me_archive
        self.an_me_archive = an_me_archive
        self.genotypeDivEvaluator = genotypeDivEvaluator
        self.actual_generation = 1

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
        self.indicator_set = [
            "id",
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
        
        # # We are going to have three MAP-Elites archives, for all we are going to store a 2-vector (Fitness, Unaligned Novelty) in each bin, for analysis purposes
        # min_max_gr = [(0, self.total_voxels, self.total_voxels + 1), (0, self.total_voxels, self.total_voxels + 1)]
        # lower_bound = np.array([0,0])
        # upper_bound = np.array([self.total_voxels, self.total_voxels])
        # ppd = np.array([self.total_voxels + 1, self.total_voxels + 1])
        # #1.- Elites in terms of fitness
        # self.f_me_archive = MAP_ElitesArchive("f_elites", self.json_base_path, lower_bound, upper_bound, ppd, self.extract_morpho, bins_type=int)
        # #2.- Elites in terms of aligned novelty
        # self.an_me_archive = MAP_ElitesArchive("an_elites", self.json_base_path, lower_bound, upper_bound, ppd, self.extract_morpho, bins_type=int)

        self.checkpoint_path = os.path.join(self.json_base_path, f"analytics_checkpoint.pickle")

    def start(self, **kwargs):
        resuming_run = kwargs["resuming_run"]
        isNewExperiment = kwargs["isNewExperiment"]

        # self.f_me_archive.start(resuming_run = resuming_run)
        # self.an_me_archive.start(resuming_run = resuming_run)

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
        pop = [ind.X[0] for ind in kwargs['pop']]
        # child_pop = kwargs['child_pop']
        
        self.init_indicator_mapping()

        # Compute aligned novelty scores
        an_scores = self.an_archive.evaluation_fn(pop, **kwargs)
        # Assign aligned novelty scores
        for i, individual in enumerate(pop):
            individual.aligned_novelty = -an_scores[i]
        if issubclass(type(problem), BaseSingleObjectiveProblem) or issubclass(type(problem), BaseMEProblem):
            # Compute unaligned novelty scores
            un_scores = self.un_archive.evaluation_fn(pop, **kwargs)
            # Assign unaligned novelty scores
            for i, individual in enumerate(pop):
                individual.unaligned_novelty = -un_scores[i]

        # Assign novelty scores to archive objects
        # unaligned novelty archive
        for i, individual in enumerate(self.un_archive.novelty_archive):
            individual.unaligned_novelty = self.un_archive.novelty_scores[i]
            self.indicator_mapping["unaligned_novelty_archive_novelty"] += [individual.unaligned_novelty]
            self.indicator_mapping["unaligned_novelty_archive_fit"] += [individual.fitness]
        # aligned novelty archive
        for i, individual in enumerate(self.an_archive.novelty_archive):
            individual.aligned_novelty = self.an_archive.novelty_scores[i]
            self.indicator_mapping["aligned_novelty_archive_novelty"] += [individual.aligned_novelty]
            self.indicator_mapping["aligned_novelty_archive_fit"] += [individual.fitness]

        self.an_me_archive.update_existing_archive(self.an_archive, "aligned_novelty", "aligned_novelty")
        
        self.an_me_archive.update_existing_archive(self.un_archive, "unaligned_novelty", "aligned_novelty")

        self.f_me_archive.update_existing_archive(self.an_archive, "aligned_novelty", "fitness")

        self.f_me_archive.update_existing_archive(self.un_archive, "unaligned_novelty", "fitness")

        for ind in pop:
            if not issubclass(type(algorithm.survival), MESurvival): 
                self.f_me_archive.try_add(ind, -ind.fitness)
            self.an_me_archive.try_add(ind, -ind.aligned_novelty)

        if not algorithm.is_initialized:
            return

        for ind in pop:
            endpoint = self.extract_endpoint(ind)
            inipoint = self.extract_initpoint(ind)
            trayectory = self.extract_trayectory(ind)
            morphology = self.extract_morpho(ind)
            self.indicator_mapping["id"] += [ind.id]
            self.indicator_mapping["generation"] += [self.actual_generation]
            self.indicator_mapping["run"] += [self.run]
            self.indicator_mapping["md5"] += [ind.md5]
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
            self.indicator_mapping["fitness"] += [ind.fitness]
            self.indicator_mapping["unaligned_novelty"] += [ind.unaligned_novelty]
            self.indicator_mapping["aligned_novelty"] += [ind.aligned_novelty]
            self.indicator_mapping["gene_diversity"] += [ind.gene_diversity]
            self.indicator_mapping["control_gene_div"] += [ind.control_gene_div]
            self.indicator_mapping["morpho_gene_div"] += [ind.morpho_gene_div]
            control_cppn = ind.genotype.networks[0].graph
            morpho_cppn = ind.genotype.networks[1].graph
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
        self.indicator_mapping["coverage"] += [self.f_me_archive.coverage()]
        f_qd_scores = self.f_me_archive.qd_scores({'fitness':'qd-score_ff', 'unaligned_novelty':'qd-score_fun', 'aligned_novelty':'qd-score_fan'})
        an_qd_scores = self.an_me_archive.qd_scores({'fitness':'qd-score_anf', 'unaligned_novelty':'qd-score_anun', 'aligned_novelty':'qd-score_anan'})

        for score in f_qd_scores.keys():
            self.indicator_mapping[score] += [f_qd_scores[score]]

        for score in an_qd_scores.keys():
            self.indicator_mapping[score] += [an_qd_scores[score]]

        indicator_df = self.indicator_df()
        indicator_df.to_csv(self.indicators_csv_path, mode='a', header=not os.path.exists(self.indicators_csv_path), index = False)

        stats_df = self.indicator_stats_df()
        stats_df.to_csv(self.stats_csv_path, mode='a', header=not os.path.exists(self.stats_csv_path), index = False)

        
    def save_archives(self, algorithm):
        problem = algorithm.problem
        archives = {
            "f_me_archive" : [],
            "an_me_archive" : [],
            "novelty_archive_un" : [],
            "novelty_archive_an" : []
        }
        if issubclass(type(algorithm.survival), MESurvival):
            self.f_me_archive = algorithm.survival.me_archive
        if "unaligned_novelty" in problem.evaluators:
            unaligned_archive_key = "unaligned_novelty"
        elif "unaligned_nslc" in problem.evaluators:
            unaligned_archive_key = "unaligned_nslc"

        an_novelty_archive = problem.evaluators["aligned_novelty"]
        un_novelty_archive = problem.evaluators[unaligned_archive_key]

        # Coverage is the same for all archives
        for i in range(len(self.f_me_archive)):
            xf = self.f_me_archive[i]
            xan = self.an_me_archive[i]
            # If one is None, all are None
            if xf is not None:
                archives["f_me_archive"] += [[xf.md5, xf.id, xf.fitness, xf.unaligned_novelty, xf.aligned_novelty]]
                archives["an_me_archive"] += [[xan.md5, xan.id, xan.fitness, xan.unaligned_novelty, xan.aligned_novelty]]
                # saveToPickle(f"{self.f_me_archive.archive_path}/elite_{i}.pickle", xf)
                # saveToPickle(f"{self.an_me_archive.archive_path}/elite_{i}.pickle", xan)
            else:
                archives["f_me_archive"] += [0]
                archives["an_me_archive"] += [0]

        for xan in an_novelty_archive.novelty_archive:
            archives["novelty_archive_an"] += [[xan.md5, xan.id, xan.fitness, xan.unaligned_novelty, xan.aligned_novelty]]
        
        for xun in un_novelty_archive.novelty_archive:
            archives["novelty_archive_un"] += [[xun.md5, xun.id, xun.fitness, xun.unaligned_novelty, xun.aligned_novelty]]

        
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
        

