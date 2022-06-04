
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

    def vector_field_distance(self, ind1, ind2, indexes, output_tags):
        gene_length = len(ind1.genotype)
        avg_dist = 0
        for i,j,k in indexes:
            dist = 0
            for gene_index in range(gene_length):
                g1 = ind1.genotype[gene_index].graph
                g2 = ind2.genotype[gene_index].graph
                v1 = []
                v2 = []
                # Form output vectors
                for output_name in output_tags[gene_index]:
                    v1 += [g1.nodes[output_name]["state"][i, j, k]]
                    v2 += [g2.nodes[output_name]["state"][i, j, k]]

                v1 = np.array(v1)
                v2 = np.array(v2)

                v1_norm = np.sqrt(np.sum(v1**2))
                v2_norm = np.sqrt(np.sum(v2**2))

                cos_sim = np.dot(v1, v2)/(v1_norm*v2_norm)

                # euclidean_dist = np.exp(-np.sqrt(np.sum((p1 - p2)**2)))
                angle_sim = np.arccos(max(-1., min(1., cos_sim)))/np.pi
                magn_sim = abs(v1_norm - v2_norm)

                dist += 1/3*(angle_sim + magn_sim)

            avg_dist += dist/gene_length

        avg_dist /= len(indexes)

        return avg_dist

    @timeit
    def _do(self, problem, pop, *args, n_survive=None, **kwargs):

        #if not (self.input_tags or self.output_tags):
        if not self.io_tags_cached:            
            for net in pop[0].X.genotype:
                self.input_tags += [set()]
                self.output_tags += [set()]
                for name in net.graph.nodes:
                    if net.graph.nodes[name]['type'] == 'input':
                        self.input_tags[-1].add(name)
                    elif net.graph.nodes[name]['type'] == 'output':
                        self.output_tags[-1].add(name)
            self.io_tags_cached = True

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



        logger.debug("Starting vector field distance calculation in parallel...")
        dist_dict = {}
        with concurrent.futures.ProcessPoolExecutor() as executor:
            future_to_indexes = {}
            for i in range(len(fronts_indxs)):              
                for j in range(i + 1, len(fronts_indxs)):
                    row_indx, col_indx = fronts_indxs[i], fronts_indxs[j]
                    future_to_indexes[executor.submit(self.vector_field_distance, pop[row_indx].X, pop[col_indx].X, self.indexes, self.output_tags)] = (row_indx,col_indx)

            for future in concurrent.futures.as_completed(future_to_indexes):
                row_indx, col_indx = future_to_indexes[future]
                dist_dict[(row_indx, col_indx)] = future.result()
        logger.debug("Finished vector field distance calculation...")


        for k, front in enumerate(fronts):
            
            idxs = []
            for i, f in enumerate(fronts):
                if i != k:
                    idxs += list(f)
            
            # calculate the divesity of the front
            diversity_of_front = vector_field_diversity(front, np.array(idxs), len(fronts_indxs), dist_dict)
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


def vector_field_diversity(current_front, other_fronts, len_fronts, dist_dict):

    pop_diversity = []

    for i in current_front:
        distance_sum = 0
        
        for j in current_front:
            if i != j:
                distance_sum += dist_dict[(i,j)] if (i, j) in dist_dict else dist_dict[(j, i)] 



        for k in other_fronts:
            distance_sum += dist_dict[(i, k)] if (i, k) in dist_dict else dist_dict[(k, i)] 
                
        distance_sum /= len_fronts
        pop_diversity += [distance_sum]
    
    return np.array(pop_diversity)





                

