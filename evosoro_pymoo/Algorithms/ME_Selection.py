import math
import random

import numpy as np

from pymoo.core.selection import Selection
from pymoo.util.misc import random_permuations
from pymoo.algorithms.moo.nsga2 import binary_tournament

from evosoro_pymoo.Algorithms.MAP_Elites import MAP_ElitesArchive



class MESelection(Selection):

    def __init__(self, me_archive : MAP_ElitesArchive) -> None:
        self.me_archive = me_archive
        self.f_comp = binary_tournament

    def _do(self, pop, n_select, n_parents=1, **kwargs):

        # Get indexes of filled bins
        filled_bins_indxs = [indx for indx, val in enumerate(self.me_archive.filled_elites_archive) if val >=0.5]

        # Uniformly select filled bins
        selected_bins = random.sample(filled_bins_indxs, n_select*n_parents)
        for i, bin in enumerate(selected_bins):
            pop[i].X = self.me_archive[bin]

        return np.arange(0,n_select*n_parents).reshape((n_select, n_parents))