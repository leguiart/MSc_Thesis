
import numpy as np
import math
import networkx as nx
import itertools
import concurrent.futures
import logging

from pymoo.core.survival import Survival
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.randomized_argsort import randomized_argsort
from common.Utils import timeit


logger = logging.getLogger(f"__main__.{__name__}")
np.seterr(divide='ignore', invalid='ignore')
# ---------------------------------------------------------------------------------------------------------
# Survival Selection
# ---------------------------------------------------------------------------------------------------------

class RankAndVectorFieldDiversitySurvival(Survival):

    def __init__(self, nds=None, orig_size_xyz = (6,6,6)) -> None:
        super().__init__(filter_infeasible=True)
        self.nds = nds if nds is not None else NonDominatedSorting()
        ranges = [list(range(orig_size_xyz[0])), list(range(orig_size_xyz[1])), list(range(orig_size_xyz[2]))]
        self.indexes = [list(element) for element in itertools.product(*ranges)]
        self.input_tags = []
        self.output_tags = []
        self.io_tags_cached = False

    @timeit
    def _do(self, problem, pop, *args, n_survive=None, **kwargs):
        

        # get the objective space values and objects
        F = pop.get("F").astype(float, copy=False)

        # the final indices of surviving individuals
        survivors = []

        # do the non-dominated sorting until splitting front
        fronts = self.nds.do(F, n_stop_if_ranked=n_survive)

        # Calculate average vector field distance between each individual
        fronts_indxs = []
        max = -math.inf
        for front in fronts:
            for indx in front:
                if indx > max:
                    max = indx
                fronts_indxs += [indx]

        population_of_fronts = [pop[indx].X for indx in fronts_indxs]
        population_parent_and_child = [individual.X for individual in pop]

        problem.evaluators["genotype_distance_evaluator"].evaluate(population_of_fronts)
        dist_dict = problem.evaluators["genotype_distance_evaluator"].distance_cache

        for k, front in enumerate(fronts):
            
            idxs = []
            for i, f in enumerate(fronts):
                if i != k:
                    idxs += list(f)
            
            # calculate the divesity of the front
            diversity_of_front = vector_field_diversity(front, fronts_indxs, population_parent_and_child, dist_dict)
            # save rank in the individual class
            for j, i in enumerate(front):
                pop[i].set("rank", k)
                pop[i].set("crowding", diversity_of_front[j])

            # current front sorted by diversity if splitting
            if len(survivors) + len(front) > n_survive:

                I = randomized_argsort(diversity_of_front, order='descending', method='numpy')
                I = I[:(n_survive - len(survivors))]

            # otherwise take the whole front unsorted
            else:
                I = np.arange(len(front))

            # extend the survivors by all or selected individuals
            survivors.extend(front[I])

        return pop[survivors]


def vector_field_diversity(front_indxs, fronts_indxs, pop, dist_dict):
    front_diversity = []

    for indx1 in front_indxs:
        distance_sum = 0
        for indx2 in fronts_indxs:
            if indx1 != indx2:
                ind1_md5 = pop[indx1].md5
                ind2_md5 = pop[indx2].md5
                if (ind1_md5, ind2_md5) in dist_dict:
                    distance_sum += dist_dict[(pop[indx1].md5, pop[indx2].md5)]
                elif (ind2_md5, ind1_md5) in dist_dict:
                    distance_sum += dist_dict[(pop[indx2].md5, pop[indx1].md5)] 

        front_diversity += [distance_sum/len(fronts_indxs)]
    return np.array(front_diversity)



    # pop_diversity = []

    # for i in current_front:
    #     distance_sum = 0
        
    #     for j in current_front:
    #         if i != j:
    #             distance_sum += dist_dict[(i,j)] if (i, j) in dist_dict else dist_dict[(j, i)] 



    #     for k in other_fronts:
    #         distance_sum += dist_dict[(i, k)] if (i, k) in dist_dict else dist_dict[(k, i)] 
                
    #     distance_sum /= len_fronts
    #     pop_diversity += [distance_sum]
    
    # return np.array(pop_diversity)





def vector_field_distance(ind1, ind2, indexes, output_tags):
    gene_length = len(ind1.genotype)
    avg_dist = 0
    
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
        avg_dist += np.mean(dist)

        # for indx, triplet in enumerate(indexes):
        #     i,j,k  = triplet
        #     dist = 0
        
        #     g1 = ind1.genotype[gene_index].graph
        #     g2 = ind2.genotype[gene_index].graph
        #     v1 = []
        #     v2 = []
        #     # Form output vectors
        #     for output_name in output_tags[gene_index]:
        #         v1 += [g1.nodes[output_name]["state"][i, j, k]]
        #         v2 += [g2.nodes[output_name]["state"][i, j, k]]
            
        #     v1 = np.array(v1)
        #     v2 = np.array(v2)
        #     vectors1[indx] = v1
        #     vectors2[indx] = v2

            # v1_norm = np.sqrt(np.sum(v1**2))
            # v2_norm = np.sqrt(np.sum(v2**2))

            # cos_sim = np.dot(v1, v2)/(v1_norm*v2_norm)
            # cos_sim_normalized = (cos_sim + 1)/2
            # cos_dist = 1 - cos_sim_normalized

            # # euclidean_dist = np.exp(-np.sqrt(np.sum((p1 - p2)**2)))
            # # angle_sim = np.arccos(max(-1., min(1., cos_sim)))/np.pi
            # magn_sim = abs(v1_norm - v2_norm)

            # dist += 1/2*(cos_dist + magn_sim)
        # v1_norm = np.sqrt(np.sum(vectors1**2, axis=1))
        # v2_norm = np.sqrt(np.sum(vectors2**2, axis=1))
        # cos_sim = (np.sum(vectors1 * vectors2, axis = 1))/(v1_norm*v2_norm)
        # cos_sim_normalized = (cos_sim + 1)/2
        # cos_dist = 1 - cos_sim_normalized

        # magn_sim = np.abs(v1_norm - v2_norm)
        # magn_sim_normalized = np.nan_to_num((magn_sim - np.min(magn_sim))/(np.max(magn_sim) - np.min(magn_sim)), nan=1.)
        # magn_dist = 1 - magn_sim_normalized

        # dist = 1/2*(cos_dist + magn_dist)
        # avg_dist += np.mean(dist)

    avg_dist /= gene_length

    return avg_dist            

