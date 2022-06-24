
from collections import _KT, _VT
import concurrent.futures
import logging
from typing import Dict, List, Set, Tuple
import numpy as np
from evosoro.softbot import SoftBot

from evosoro_pymoo.Evaluators.IEvaluator import IEvaluator

logger = logging.getLogger(f"__main__.{__name__}")
np.seterr(divide='ignore', invalid='ignore')

class GenotypeDistanceEvaluator(IEvaluator, dict):

    def __init__(self) -> None:
        super().__init__()
        self.distance_cache : Dict[Tuple[str, str], List[float]] = {}
        self.input_tags : List[Set[str]] = []
        self.output_tags : List[Set[str]]= []
        self.io_tags_cached : bool = False


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


    def evaluate(self, X : List[SoftBot], *args, **kwargs) -> List[SoftBot]:
        if not self.io_tags_cached:            
            for net in X[0].genotype:
                self.input_tags += [set()]
                self.output_tags += [set()]
                for name in net.graph.nodes:
                    if net.graph.nodes[name]['type'] == 'input':
                        self.input_tags[-1].add(name)
                    elif net.graph.nodes[name]['type'] == 'output':
                        self.output_tags[-1].add(name)
            self.io_tags_cached = True

        logger.debug("Starting vector field distance calculation...")

        with concurrent.futures.ProcessPoolExecutor() as executor:
            future_to_indexes = {}
            for i in range(len(X)):              
                for j in range(i + 1, len(X)):
                    row_md5, col_md5 = X[i].md5, X[j].md5
                    if row_md5 != col_md5 and (row_md5, col_md5) not in self.distance_cache:
                        future_to_indexes[executor.submit(vector_field_distance, X[i], X[j], self.output_tags)] = (row_md5, col_md5)

            for future in concurrent.futures.as_completed(future_to_indexes):
                row_md5, col_md5 = future_to_indexes[future]
                self[(row_md5, col_md5)] = future.result()

        logger.debug("Finished vector field distance calculation...")

        return X



def vector_field_distance(ind1 : SoftBot, ind2 : SoftBot, output_tags : List[Set[str]]) -> List[float] :
    
    gene_length = len(ind1.genotype)
    gene_distances = []

    for gene_index in range(gene_length):
        # vectors1 = np.zeros((len(indexes), len(output_tags[gene_index])))
        # vectors2 = np.zeros((len(indexes), len(output_tags[gene_index])))

        # Form output tensors
        g1 = ind1.genotype[gene_index].graph
        g2 = ind2.genotype[gene_index].graph

        # Each graph can have multiple output nodes and each output node
        # has an associated state which contains all outputs given the design space
        # in the form of a 3D matrix.
        # We need to concatenate the matrices, and form a 4D tensor
        tensor1 = []
        tensor2 = []

        # Form output tensors
        for output_name in output_tags[gene_index]:
            tensor1 += [g1.nodes[output_name]["state"]]
            tensor2 += [g2.nodes[output_name]["state"]]

        tensor1 = np.array(tensor1).T
        tensor2 = np.array(tensor2).T
        
        t1_norm = np.sqrt(np.sum(tensor1**2, axis = 3))
        t2_norm = np.sqrt(np.sum(tensor2**2, axis = 3))
        cos_sim = (np.sum(tensor1 * tensor2, axis = 3))/(t1_norm*t2_norm)

        cos_sim_normalized = (cos_sim + 1)/2
        cos_dist = 1 - cos_sim_normalized

        magn_sim = np.abs(t1_norm - t2_norm)
        magn_sim_normalized = np.nan_to_num((magn_sim - np.min(magn_sim))/(np.max(magn_sim) - np.min(magn_sim)), nan=1.)
        magn_dist = 1 - magn_sim_normalized

        dist = 1/2*(cos_dist + magn_dist)
        gene_distances += [np.mean(dist)]


    return gene_distances     

 