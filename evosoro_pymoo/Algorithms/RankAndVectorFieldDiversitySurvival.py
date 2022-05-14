
from socket import timeout
from matplotlib.pyplot import axis
import numpy as np
import math
import networkx as nx
import itertools
from pymoo.core.survival import Survival
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.randomized_argsort import randomized_argsort

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


    def _do(self, problem, pop, *args, n_survive=None, **kwargs):

        if not (self.input_tags or self.output_tags):
            for net in pop[0].X.genotype:
                self.input_tags += [set()]
                self.output_tags += [set()]
                for name in net.graph.nodes:
                    if net.graph.nodes[name]['type'] == 'input':
                        self.input_tags[-1].add(name)
                    elif net.graph.nodes[name]['type'] == 'output':
                        self.output_tags[-1].add(name)

        # get the objective space values and objects
        F = pop.get("F").astype(float, copy=False)

        # the final indices of surviving individuals
        survivors = []

        # do the non-dominated sorting until splitting front
        fronts = self.nds.do(F, n_stop_if_ranked=n_survive)
        fronts_idxs = list(range(len(fronts)))
        len_fronts = sum([len(front) for front in fronts])
        cached_distances = {}

        for k, front in enumerate(fronts):
            
            idxs = []
            for i, f in enumerate(fronts):
                if i != k:
                    idxs += list(f)
            
            # calculate the divesity of the front
            diversity_of_front = vector_field_diversity(pop, front, np.array(idxs), len_fronts, self.indexes, self.input_tags, self.output_tags, cached_distances)
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


def vector_field_diversity(pop, current_front, other_fronts, n, indexes, input_tags, output_tags, cached_distances):

    genotype_len = len(pop[0].X.genotype)
    pop_diversity = []

    for i in current_front:
        ind1 = pop[i].X
        distance_sum = 0
        
        for j in current_front:
            ind2 = pop[j].X
            d = 0
            if i != j:
                if (i, j) not in cached_distances and (j, i) not in cached_distances:
                    for graph_idx in range(genotype_len):
                        d += vector_field_distance(ind1.genotype[graph_idx].graph, ind2.genotype[graph_idx].graph, indexes, input_tags[graph_idx], output_tags[graph_idx])
                    d /= genotype_len
                    cached_distances[(i, j)] = d
                    distance_sum += d
                    
                elif (i, j) in cached_distances:
                    distance_sum += cached_distances[(i, j)]
                elif (j, i) in cached_distances:
                    distance_sum += cached_distances[(j, i)]



        for k in other_fronts:
            ind2 = pop[k].X
            d = 0
            if (i, k) not in cached_distances and (k, i) not in cached_distances:
                for graph_idx in range(genotype_len):
                    d += vector_field_distance(ind1.genotype[graph_idx].graph, ind2.genotype[graph_idx].graph, indexes, input_tags[graph_idx], output_tags[graph_idx])
                d /= genotype_len
                cached_distances[(i, k)] = d
                distance_sum += d
                
            elif (i, k) in cached_distances:
                distance_sum += cached_distances[(i, k)]
            elif (k, i) in cached_distances:
                distance_sum += cached_distances[(k, i)]
        
        distance_sum /= n
        pop_diversity += [distance_sum]
    
    return np.array(pop_diversity)


def vector_field_distance(g1, g2, indexes, input_tags, output_tags):
    vector_distances = []
    for i,j,k in indexes:
        # for i2,j2,k2 in indexes:
        #     if i1 != i2 or j1 != j2 or k1 != k2:
        # Form input points
        # p1 = []
        # p2 = []
        # for input_name in input_tags:
        #     p1 += [g1.nodes[input_name]["state"][i1, j1, k1]]
        #     p2 += [g2.nodes[input_name]["state"][i2, j2, k2]]
        v1 = []
        v2 = []
        # Form output vectors
        for output_name in output_tags:
            v1 += [g1.nodes[output_name]["state"][i, j, k]]
            v2 += [g2.nodes[output_name]["state"][i, j, k]]
        # p1 = np.array(p1)
        # p2 = np.array(p2)
        v1 = np.array(v1)
        v2 = np.array(v2)

        v1_norm = np.sqrt(np.sum(v1**2))
        v2_norm = np.sqrt(np.sum(v2**2))

        # euclidean_dist = np.exp(-np.sqrt(np.sum((p1 - p2)**2)))
        angle_dist = np.exp(1.-max(-1.0, min(np.dot(v1, v2)/(v1_norm*v2_norm), 1.0)))
        magn_diff = np.exp(-abs(v1_norm - v2_norm))

        d = 1/3*(1 + angle_dist + magn_diff)

        vector_distances += [1 - d]

    return abs(np.mean(vector_distances))


                

