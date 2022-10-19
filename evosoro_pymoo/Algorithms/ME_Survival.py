
import numpy as np
from sklearn.neighbors import KDTree
from pymoo.core.survival import Survival
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.randomized_argsort import randomized_argsort
from pymoo.algorithms.moo.nsga2 import RankAndCrowdingSurvival

from evosoro_pymoo.Algorithms.MAP_Elites import MAP_ElitesArchive

# ---------------------------------------------------------------------------------------------------------
# Survival Selection
# ---------------------------------------------------------------------------------------------------------


class MESurvival(Survival):

    def __init__(self, me_archive : MAP_ElitesArchive) -> None:
        super().__init__(filter_infeasible=True)
        self.me_archive = me_archive

    def _do(self, problem, pop, *args, n_survive=None, **kwargs):
        obj_pop = [x_i.X for x_i in pop] if n_survive == len(pop) else [x_i.X for x_i in pop[n_survive:]]
        indexes = []
        for i, ind in enumerate(obj_pop):
            if self.me_archive.try_add(ind):
                indexes += [i]

        return pop[np.arange(0,n_survive)]