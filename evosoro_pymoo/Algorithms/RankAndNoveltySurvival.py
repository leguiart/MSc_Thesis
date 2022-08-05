
import numpy as np
from sklearn.neighbors import KDTree
from pymoo.core.survival import Survival
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.randomized_argsort import randomized_argsort
from pymoo.algorithms.moo.nsga2 import RankAndCrowdingSurvival

# ---------------------------------------------------------------------------------------------------------
# Survival Selection
# ---------------------------------------------------------------------------------------------------------


class RankAndNoveltySurvival(Survival):

    def __init__(self, nds=None) -> None:
        super().__init__(filter_infeasible=True)
        self.nds = nds if nds is not None else NonDominatedSorting()

    def _do(self, problem, pop, *args, n_survive=None, **kwargs):

        # get the objective space values and objects
        F = pop.get("F").astype(float, copy=False)

        # the final indices of surviving individuals
        survivors = []

        # do the non-dominated sorting until splitting front
        fronts = self.nds.do(F, n_stop_if_ranked=n_survive)

        for k, front in enumerate(fronts):

            # calculate the novelty of the front
            novelty_of_front = get_unaligned_novelty(pop[front])

            # save rank and crowding in the individual class
            for j, i in enumerate(front):
                pop[i].set("rank", k)
                pop[i].set("crowding", novelty_of_front[j])

            # current front sorted by novelty if splitting
            if len(survivors) + len(front) > n_survive:
                I = randomized_argsort(novelty_of_front, order='descending', method='numpy')
                I = I[:(n_survive - len(survivors))]

            # otherwise take the whole front unsorted
            else:
                I = np.arange(len(front))

            # extend the survivors by all or selected individuals
            survivors.extend(front[I])

        return pop[survivors]


def get_unaligned_novelty(pop):
    return np.array([x_i.X.unaligned_novelty for x_i in pop])


class RankAndCrowdingNoveltySurvival(RankAndCrowdingSurvival):
    def __init__(self, nds=None) -> None:
        super().__init__(nds=nds)

    def _do(self, problem, pop, *args, n_survive=None, **kwargs):
        # Update parent population novelty
        if len(pop) > n_survive:
            n_parents = len(pop) // 2
            parent_pop = [individual.X for individual in pop[:n_parents]]
            children_pop = [individual.X for individual in pop[n_parents:]]
            novelty_evaluator = problem.evaluators["unaligned_nslc"]

            union_hashtable = {}
            union_matrix = []
            union_fitness_matrix = []

            # Compute the Union of Novelty archive and the population
            for individual in children_pop + novelty_evaluator.novelty_archive:
                if individual.md5 not in union_hashtable:
                    union_hashtable[individual.md5] = individual
                    union_matrix += [novelty_evaluator.vector_extractor(individual)]
                    union_fitness_matrix += [getattr(individual, novelty_evaluator.fitness_name)]
            
            objectives_mat = []
            constraints_mat = []
            union_matrix = np.array(union_matrix)
            kd_tree = KDTree(union_matrix)
            novelty_metric = novelty_evaluator.novelty_name
            nslc_quality_name = novelty_evaluator.nslc_quality_name

            for parent in parent_pop:
                if parent.md5 in union_hashtable:
                    updated_novelty = getattr(union_hashtable[parent.md5], novelty_metric)
                    updated_nslc_quality = getattr(union_hashtable[parent.md5], nslc_quality_name)
                else:
                    distances, kn_neighbors_indx = kd_tree.query([novelty_evaluator.vector_extractor(parent)], min(novelty_evaluator.k_neighbors, len(union_matrix)))
                    updated_novelty = np.mean(distances)
                    # updated_novelty, kn_neighbors_indx = novelty_evaluator._average_knn_distance([novelty_evaluator.vector_extractor(parent)], kd_tree)
                    setattr(X[i], self.nslc_quality_name, self.k_neighbors - sum([1 if getattr(X[i], self.fitness_name) < f else 0 for f in kn_neighborsf]))
                setattr(parent, novelty_metric, updated_novelty)
                objectives_mat += [problem._extractObjectives(parent)]
                constraint = problem._extractConstraints(parent)
                constraints_mat += [constraint] if constraint else [[0]]
            
            F = np.array(objectives_mat, dtype=float)
            G = np.array(constraints_mat, dtype=float)
            pop[:n_parents].set("F", F)
            pop[:n_parents].set("G", G)
        return super()._do(problem, pop, *args, n_survive=n_survive, **kwargs)