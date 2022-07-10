
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

        problem.evaluators["genotype_diversity_evaluator"].genotypeDistanceEvaluator.evaluate(population_of_fronts)
        dist_dict = problem.evaluators["genotype_diversity_evaluator"].genotypeDistanceEvaluator

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
                distance_sum += np.mean(dist_dict[(ind1_md5, ind2_md5)])

        front_diversity += [distance_sum/len(fronts_indxs)]
    return np.array(front_diversity)


