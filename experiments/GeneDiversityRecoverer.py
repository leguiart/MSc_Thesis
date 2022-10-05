
import glob
import os
import sys
import concurrent.futures
import logging
import numpy as np
import pandas as pd

from dotenv import load_dotenv
from typing import Dict, List, Set, Tuple


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# create file handler which logs only warning level messages
fh = logging.FileHandler('diversity.log')
fh.setLevel(logging.WARNING)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)



load_dotenv()
sys.path.append(os.getenv('PYTHONPATH'))# Appending repo's root dir in the python path to enable subsequent imports

from experiments.Constants import *
from common.Utils import readFromPickle, timeit, writeToJson
from evosoro_pymoo.Evaluators.IEvaluator import IEvaluator
from evosoro.softbot import Genotype, Phenotype


np.seterr(divide='ignore', invalid='ignore')

class GenotypeDistanceEvaluator(IEvaluator, dict):

    def __init__(self, orig_size_xyz = (6,6,6)) -> None:
        super().__init__()
        self.distance_cache : Dict[Tuple[str, str], List[float]] = {}
        self.input_tags : List[Set[str]] = []
        self.output_tags : List[Set[str]]= []
        self.io_tags_cached : bool = False
        self.dx = 0
        self.dy = 0
        self.dz = 0
        self.size_xyz = orig_size_xyz


    def __getitem__(self, __k: Tuple[int, int]) -> List[float]:
        if (__k[0],__k[1]) in self.distance_cache:
            return self.distance_cache[(__k[0],__k[1]) ]
        elif (__k[1],__k[0]) in self.distance_cache:
            return self.distance_cache[(__k[1],__k[0])]
        return [0. for _ in range(len(self.input_tags))]


    def __setitem__(self, __k : Tuple[int, int], __v: List[float]) -> None:
        self.distance_cache[__k] = __v


    def __contains__(self, __o: Tuple[int, int]) -> bool:
        return (__o[0], __o[1]) not in self.distance_cache and (__o[1], __o[0]) not in self.distance_cache

    @timeit
    def evaluate(self, X : List[Tuple[int, Genotype]], *args, **kwargs) -> None:
        if not self.io_tags_cached:            
            for net in X[0][1]:
                self.input_tags += [set()]
                self.output_tags += [set()]
                for name in net.graph.nodes:
                    if net.graph.nodes[name]['type'] == 'input':
                        if name == 'x':
                            self.dx = (np.max(net.graph.nodes[name]['state']) - np.min(net.graph.nodes[name]['state'])) / self.size_xyz[0]
                        elif name == 'y':
                            self.dy = (np.max(net.graph.nodes[name]['state']) - np.min(net.graph.nodes[name]['state'])) / self.size_xyz[1]
                        elif name == 'z':
                            self.dz = (np.max(net.graph.nodes[name]['state']) - np.min(net.graph.nodes[name]['state'])) / self.size_xyz[2]
                        self.input_tags[-1].add(name)
                    elif net.graph.nodes[name]['type'] == 'output':
                        self.output_tags[-1].add(name)
            self.io_tags_cached = True

        logger.debug("Starting vector field distance calculation...")

        dxdydz = self.dx*self.dy*self.dz
        # with concurrent.futures.ProcessPoolExecutor() as executor:
        #     future_to_indexes = {}
        #     for i in range(len(X)):              
        #         for j in range(i + 1, len(X)):
        #             row_id, col_id = X[i][0], X[j][0]
        #             if row_id != col_id and ((row_id, col_id) not in self.distance_cache or (col_id, row_id) not in self.distance_cache):
        #                 future_to_indexes[executor.submit(vector_field_distance, X[i][1], X[j][1], self.output_tags, dxdydz)] = (row_id, col_id)

        #     for future in concurrent.futures.as_completed(future_to_indexes):
        #         row_id, col_id = future_to_indexes[future]
        #         self[(row_id, col_id)] = future.result()

        for i in range(len(X)):              
            for j in range(i + 1, len(X)):
                row_id, col_id = X[i][0], X[j][0]
                if row_id != col_id and ((row_id, col_id) not in self.distance_cache or (col_id, row_id) not in self.distance_cache):
                    self[(row_id, col_id)] = vector_field_distance(X[i][1], X[j][1], self.output_tags, dxdydz)

        logger.debug("Finished vector field distance calculation...")




def vector_field_distance(gene1 : Genotype, gene2 : Genotype, output_tags : List[Set[str]], dxdydz : float) -> List[float] :
    
    gene_length = len(gene1)
    gene_distances = []

    # Distances between each chromosome
    for gene_index in range(gene_length):

        g1 = gene1[gene_index].graph
        g2 = gene2[gene_index].graph
        matrix1 = []
        matrix2 = []

        for output_name in output_tags[gene_index]:
            matrix1 += [g1.nodes[output_name]["state"].flatten()]
            matrix2 += [g2.nodes[output_name]["state"].flatten()]

        # tensor1 = np.array(tensor1).T
        # tensor2 = np.array(tensor2).T
        
        matrix1 = np.array(matrix1)
        matrix2 = np.array(matrix2)

        matrix_diff = matrix1 - matrix2
        vect_dist = np.sum(matrix_diff**2, axis=0)*dxdydz


        gene_distances += [np.sqrt(np.sum(vect_dist))]
    
    matrix1 = []
    matrix2 = []
    # Distances between all chromosomes
    for gene_index in range(gene_length):

        g1 = gene1[gene_index].graph
        g2 = gene2[gene_index].graph


        for output_name in output_tags[gene_index]:
            matrix1 += [g1.nodes[output_name]["state"].flatten()]
            matrix2 += [g2.nodes[output_name]["state"].flatten()]
        
    matrix1 = np.array(matrix1)
    matrix2 = np.array(matrix2)

    tensor_diff = matrix1 - matrix2
    vect_dist = np.sum(tensor_diff**2, axis=0)*dxdydz


    gene_distances += [np.sqrt(np.sum(vect_dist))]


    return gene_distances    

class GenotypeDiversityEvaluator(IEvaluator, object):

    def __init__(self, orig_size_xyz = (6,6,6)) -> None:
        super().__init__()
        self.genotypeDistanceEvaluator = GenotypeDistanceEvaluator(orig_size_xyz)
        self.gene_div_lookup = {}

    def __getitem__(self, n):
        return self.gene_div_lookup[n]

    @timeit
    def evaluate(self, X : List[Tuple[int, Genotype]], *args, **kwargs) -> None:
        self.genotypeDistanceEvaluator.evaluate(X)
        self.gene_div_lookup = {}

        for i in range(len(X)):
            gene_diversity = []
            for j in range(len(X)):
                if i != j:
                    ind1_id = X[i][0]
                    ind2_id = X[j][0]
                    gene_diversity += [self.genotypeDistanceEvaluator[ind1_id, ind2_id]]
                    
            gene_diversity = np.array(gene_diversity)
            gene_diversity = np.mean(gene_diversity, axis=0)
            self.gene_div_lookup[X[i][0]] = list(gene_diversity)


def main():
    for experiment_type in ["SO", "QN", "ME", "NSLC", "MNSLC"]:
        if experiment_type != "SO":
            run_dir_prefix = f"/media/leguiart/LuisExtra/ExperimentsData2/BodyBrain{experiment_type}Data"
        else:
            run_dir_prefix = "/media/leguiart/LuisExtra/ExperimentsData2/BodyBrainData"
        
        run_dirs = glob.glob(run_dir_prefix+"*")
        div_csv_path = "/media/leguiart/LuisExtra/ExperimentsData2/gene_div.csv"
        for run_index, run_dir in enumerate(run_dirs):
            df_dict = {"Id":[], "control_div":[], "morpho_div":[], "gene_div":[], "Generation":[], "Run":[], "Method":[]}
            res_set_path = os.path.join(run_dir, "results_set.pickle")

     
            if os.path.isdir(run_dir):
                genotypeDiversityEvaluator = GenotypeDiversityEvaluator(orig_size_xyz = IND_SIZE)
                stored_bots = glob.glob(run_dir + "/Gen_*")
                gen_lst = [int(str.lstrip(str(str.split(stored_bot, '_')[-1]), '0')) for stored_bot in stored_bots]
                gen_lst.sort()
                max_gens = min(3000, len(gen_lst))
                try:
                    res_set = readFromPickle(res_set_path)
                except:
                    res_set = None

                if res_set:
                    res_pop = res_set['res'].pop
                    res_pop = [(ind.X.id, ind.X.genotype) for ind in res_pop]
                    genotypeDiversityEvaluator.evaluate(res_pop)
                    for id, _ in res_pop:
                        diversity = genotypeDiversityEvaluator[id]
                        df_dict["Id"] += [id]
                        df_dict["control_div"] += [diversity[0]]
                        df_dict["morpho_div"] += [diversity[1]]
                        df_dict["gene_div"] += [diversity[2]]
                        df_dict["Generation"] += [max_gens]
                        df_dict["Run"] += [run_index + 1]
                        df_dict["Method"] += [experiment_type]

                pd.DataFrame(df_dict).to_csv(div_csv_path, mode='a', header=not os.path.exists(div_csv_path), index = False)
                run_not_included = False

                for gen in gen_lst[:max_gens]:
                    gen_path = os.path.join(run_dir, f"Gen_{gen:04d}")
                    if os.path.isdir(gen_path):
                        nn_backup_path = os.path.join(gen_path, f"Gen_{gen:04d}_networks.pickle")
                        
                        try:
                            generation_nns = readFromPickle(nn_backup_path)
                        except:
                            generation_nns = None
                        if generation_nns:
                            genotypeDiversityEvaluator.evaluate(generation_nns)
                            parents = generation_nns[:len(generation_nns)//2]
                            
                            for id, _ in parents:
                                diversity = genotypeDiversityEvaluator[id]
                                df_dict["Id"] += [id]
                                df_dict["control_div"] += [diversity[0]]
                                df_dict["morpho_div"] += [diversity[1]]
                                df_dict["gene_div"] += [diversity[2]]
                                df_dict["Generation"] += [gen]
                                df_dict["Run"] += [run_index + 1]
                                df_dict["Method"] += [experiment_type]
                        else:
                            run_not_included = True
                            break
                


                

                            

if __name__ == "__main__":
    main()
