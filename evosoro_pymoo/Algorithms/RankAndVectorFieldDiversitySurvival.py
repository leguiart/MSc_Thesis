
import numpy as np
import math
import logging
from pymoo.core.survival import Survival
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.randomized_argsort import randomized_argsort


from utils.utils import timeit
from evosoro_pymoo.Evaluators.GenotypeDiversityEvaluator import GenotypeDiversityEvaluator


logger = logging.getLogger(f"__main__.{__name__}")
np.seterr(divide='ignore', invalid='ignore')
# ---------------------------------------------------------------------------------------------------------
# Survival Selection
# ---------------------------------------------------------------------------------------------------------

class RankAndVectorFieldDiversitySurvival(Survival):

    def __init__(self, genotypeDiversityEvaluator : GenotypeDiversityEvaluator, nds=None) -> None:
        super().__init__(filter_infeasible=True)
        self.genotypeDiversityEvaluator = genotypeDiversityEvaluator
        self.nds = nds if nds is not None else NonDominatedSorting()
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

        dist_dict = self.genotypeDiversityEvaluator.genotypeDistanceEvaluator
        dist_dict.evaluate(population_of_fronts)
        
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
                ind1_id = pop[indx1][0].id
                ind2_id = pop[indx2][0].id
                distance_sum += dist_dict[(ind1_id, ind2_id)][2]

        front_diversity += [distance_sum/len(fronts_indxs)]
    return np.array(front_diversity)


