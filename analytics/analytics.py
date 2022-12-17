
import os
import pickle
import numpy as np
import pandas as pd
from typing import List
from scipy.spatial import distance_matrix

from constants import *
from data.dal import Dal
from qd_pymoo.Algorithm.ME_Survival import MESurvival
from evosoro_pymoo.common.IAnalytics import IAnalytics
from qd_pymoo.Problems.ME_Problem import BaseMEProblem
from qd_pymoo.Algorithm.ME_Archive import MAP_ElitesArchive
from qd_pymoo.Evaluators.NoveltyEvaluator import NoveltyEvaluatorKD
from qd_pymoo.Problems.SingleObjectiveProblem import BaseSingleObjectiveProblem
from evosoro_pymoo.Evaluators.GenotypeDiversityEvaluator import GenotypeDiversityEvaluator
from utils.utils import readFromPickle, saveToPickle, timeit, flatten_cppn_outputs


class QD_Analytics(IAnalytics):
    def __init__(self, run_id, method, experiment_name, experiment_id, json_base_path, csv_base_path,
                un_archive : NoveltyEvaluatorKD, an_archive : NoveltyEvaluatorKD, 
                f_me_archive : MAP_ElitesArchive, an_me_archive : MAP_ElitesArchive, 
                genotypeDivEvaluator : GenotypeDiversityEvaluator,
                dal : Dal,
                stored_cppns_cache : set):
        super().__init__()
        self.run_id = run_id
        self.method = method
        self.init_paths(experiment_name, json_base_path, csv_base_path)
        self.experiment_id = experiment_id
        self.un_archive = un_archive
        self.an_archive = an_archive
        self.f_me_archive = f_me_archive
        self.an_me_archive = an_me_archive
        self.genotypeDivEvaluator = genotypeDivEvaluator
        self.dal = dal
        self.stored_cppns_cache = stored_cppns_cache
        # self.actual_generation = 1

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
            "individual_id",
            "generation",
            "run_id",
            "population_type",
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
        
        self.cppn_outputs_set = [
            "experiment_id",
            "run_id",
            "md5",
            "cppn_outputs"
        ]

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
            "individual_id" : [],
            "generation" : [],
            "population_type" : [],
            "run_id" : [],
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
        actual_generation = algorithm.n_gen
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

        self.genotypeDivEvaluator.evaluate(pop)

        cppn_outputs = dict(zip(self.cppn_outputs_set, [[] for _ in self.cppn_outputs_set]))

        for i, ind in enumerate(pop):
            endpoint = self.extract_endpoint(ind)
            inipoint = self.extract_initpoint(ind)
            trayectory = self.extract_trayectory(ind)
            morphology = self.extract_morpho(ind)
            self.indicator_mapping["individual_id"] += [ind.id]
            self.indicator_mapping["generation"] += [int(actual_generation)]
            self.indicator_mapping["population_type"] += ["child" if i >= len(pop) // 2 else "parent"]
            # self.indicator_mapping["run"] += [self.run]
            self.indicator_mapping["run_id"] += [self.run_id]
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
            self.indicator_mapping["morphology_active"] += [int(morphology[0])]
            self.indicator_mapping["morphology_passive"] += [int(morphology[1])]
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
            if ind.md5 not in self.stored_cppns_cache:
                self.stored_cppns_cache.add(ind.md5)
                cppn_outputs["experiment_id"]+=[self.experiment_id]
                cppn_outputs["run_id"]+=[self.run_id]
                cppn_outputs["md5"] += [ind.md5]
                cppn_outputs["cppn_outputs"]+=[pickle.dumps(flatten_cppn_outputs(ind, self.genotypeDivEvaluator.genotypeDistanceEvaluator.output_tags))]

        

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

        self.dal.insert_indicators(self.indicator_mapping, self.indicator_set)
        self.dal.insert_indicator_stats(self.indicator_stats(actual_generation))
        if cppn_outputs["md5"]:
            self.dal.insert_cppns(cppn_outputs)

        self.genotypeDivEvaluator.genotypeDistanceEvaluator.distance_cache = {}

        if algorithm.n_gen % 100 == 0:
            self.save_archives(algorithm.n_gen)

        # indicator_df = self.indicator_df()
        # indicator_df.to_csv(self.indicators_csv_path, mode='a', header=not os.path.exists(self.indicators_csv_path), index = False)

        # stats_df = self.indicator_stats_df()
        # stats_df.to_csv(self.stats_csv_path, mode='a', header=not os.path.exists(self.stats_csv_path), index = False)

        
    def save_archives(self, generation):
        self.dal.insert_experiment_archives(self.run_id, generation, self.f_me_archive, self.an_me_archive, self.un_archive, self.an_archive)
        # save_json(self.archives_json_path, archives)


    def indicator_df(self):
        return pd.DataFrame({k : self.indicator_mapping[k] for k in self.indicator_set})
            
    def indicator_stats(self, generation):
        d = {"indicator":[], "best":[], "worst":[], "average":[], "std":[], "median":[], "generation":[], "run_id":[]}
        pointwise_indicators = ["endpoint_x", "endpoint_y", "endpoint_z", "inipoint_x", "inipoint_y", "inipoint_z", "trayectory_x", "trayectory_y", "trayectory_z"]


        for key in self.indicator_stats_set:
            # d["indicator"] += [key]
            indicatorStatsHelper(d, self.indicator_mapping, pointwise_indicators, key, 0, len(self.indicator_mapping[key]), generation, self.run_id)
            if key not in ["qd-score_ff","qd-score_fun","qd-score_fan","qd-score_anf","qd-score_anun","qd-score_anan","coverage"]:
                indicatorStatsHelper(d, self.indicator_mapping, pointwise_indicators, key, len(self.indicator_mapping[key])//2, len(self.indicator_mapping[key]), generation, self.run_id, prefix="child_")
                indicatorStatsHelper(d, self.indicator_mapping, pointwise_indicators, key, 0, len(self.indicator_mapping[key])//2, generation, self.run_id, prefix="parent_")
        return d

    def indicator_stats_df(self, generation):
        return pd.DataFrame(self.indicator_stats(generation))
        
def charToIndexHelper(c):
    if c == 'x':
        return 0
    elif c == 'y':
        return 1
    elif c == 'z':
        return 2
    else:
        raise ValueError("Invalid character passed")

def indicatorStatsHelper(d, indicator_mapping, pointwise_indicators, key, first, last, generation, run_id, prefix = ""):
    d["indicator"] += [prefix + key]
    arr = np.array(indicator_mapping[key][first:last])
    if key in pointwise_indicators:
        [indicator, coordinate_char] = key.split('_')
        coord_index = charToIndexHelper(coordinate_char)
        point_mat = np.array(indicator_mapping[indicator][first:last])
        dists = np.linalg.norm(point_mat, axis=1)

        d["best"] += [point_mat[np.argmax(dists)][coord_index]]
        d["worst"] += [point_mat[np.argmin(dists)][coord_index]]
    else:
        d["best"] += [float(np.nanmax(arr))]
        d["worst"] += [float(np.nanmin(arr))]
    d["average"] += [float(np.nanmean(arr))]
    d["std"] += [float(np.nanstd(arr))]
    d["median"] += [float(np.nanmedian(arr))]
    d["generation"] += [int(generation)]
    d["run_id"] += [run_id]