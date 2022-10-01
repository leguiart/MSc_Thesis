
import concurrent.futures
import logging
from typing import Dict, List, Set, Tuple
from matplotlib.pyplot import axis
import numpy as np
from common.Utils import timeit
from evosoro.softbot import SoftBot

from evosoro_pymoo.Evaluators.IEvaluator import IEvaluator

logger = logging.getLogger(f"__main__.{__name__}")
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


    def __getitem__(self, __k: Tuple[str, str]) -> List[float]:
        if (__k[0],__k[1]) in self.distance_cache:
            return self.distance_cache[(__k[0],__k[1]) ]
        elif (__k[1],__k[0]) in self.distance_cache:
            return self.distance_cache[(__k[1],__k[0])]
        return [0. for _ in range(len(self.input_tags))]


    def __setitem__(self, __k : Tuple[str, str], __v: List[float]) -> None:
        self.distance_cache[__k] = __v


    def __contains__(self, __o: Tuple[str, str]) -> bool:
        return (__o[0], __o[1]) not in self.distance_cache and (__o[1], __o[0]) not in self.distance_cache

    @timeit
    def evaluate(self, X : List[SoftBot], *args, **kwargs) -> List[SoftBot]:
        if not self.io_tags_cached:            
            for net in X[0].genotype:
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
        with concurrent.futures.ProcessPoolExecutor() as executor:
            future_to_indexes = {}
            for i in range(len(X)):              
                for j in range(i + 1, len(X)):
                    row_id, col_id = X[i].id, X[j].id
                    if row_id != col_id and ((row_id, col_id) not in self.distance_cache or (col_id, row_id) not in self.distance_cache):
                        future_to_indexes[executor.submit(vector_field_distance, X[i], X[j], self.output_tags, dxdydz)] = (row_id, col_id)

            for future in concurrent.futures.as_completed(future_to_indexes):
                row_id, col_id = future_to_indexes[future]
                self[(row_id, col_id)] = future.result()

        logger.debug("Finished vector field distance calculation...")

        return X



def vector_field_distance(ind1 : SoftBot, ind2 : SoftBot, output_tags : List[Set[str]], dxdydz : float) -> List[float] :
    
    gene_length = len(ind1.genotype)
    gene_distances = []

    # Distances between each chromosome
    for gene_index in range(gene_length):

        g1 = ind1.genotype[gene_index].graph
        g2 = ind2.genotype[gene_index].graph
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

        g1 = ind1.genotype[gene_index].graph
        g2 = ind2.genotype[gene_index].graph


        for output_name in output_tags[gene_index]:
            matrix1 += [g1.nodes[output_name]["state"].flatten()]
            matrix2 += [g2.nodes[output_name]["state"].flatten()]
        
    matrix1 = np.array(matrix1)
    matrix2 = np.array(matrix2)

    tensor_diff = matrix1 - matrix2
    vect_dist = np.sum(tensor_diff**2, axis=0)*dxdydz


    gene_distances += [np.sqrt(np.sum(vect_dist))]


    return gene_distances     

 




















# t1_norm = np.sqrt(np.sum(tensor1**2, axis = 3))
# t2_norm = np.sqrt(np.sum(tensor2**2, axis = 3))
# cos_sim = (np.sum(tensor1 * tensor2, axis = 3))/(t1_norm*t2_norm)
# cos_dist_gauss = np.exp(1 - cos_sim)
# cos_dist_gauss_norm = (cos_dist_gauss - 1)/(np.exp(2) - 1)

# cos_sim_normalized = (cos_sim + 1)/2
# cos_dist = 1 - cos_sim_normalized

# magn_dist_gauss = 1 - np.exp(-vect_dist)

# magn_sim = np.abs(t1_norm - t2_norm)
# magn_sim_normalized = np.nan_to_num((magn_sim - np.min(magn_sim))/(np.max(magn_sim) - np.min(magn_sim)), nan=1.)
# magn_dist = 1 - magn_sim_normalized

# dist = 1/2*(cos_dist + magn_dist)

# dist = 1/2 * (cos_dist_gauss_norm + magn_dist_gauss)

# dist = np.sqrt(np.tensordot(tensor1 - tensor2, tensor1 - tensor2, axis = 3))
# dist = np.sqrt(np.sum((tensor1 - tensor2)**2, axis = 3))