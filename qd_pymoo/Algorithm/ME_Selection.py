
import random
import numpy as np
import copy

from pymoo.core.selection import Selection
from qd_pymoo.Algorithm.ME_Archive import MAP_ElitesArchive


class MESelection(Selection):

    def __init__(self, me_archive : MAP_ElitesArchive) -> None:
        self.me_archive = me_archive

    def _do(self, pop, n_select, n_parents=1, **kwargs):
        filled_bins_indxs = self.me_archive.filled_indices
        # Uniformly select filled bins
        w = -np.array(list(filled_bins_indxs.values()))
        selected_bins = random.choices(list(filled_bins_indxs.keys()), k = min(n_select*n_parents, len(filled_bins_indxs)), weights=w)
        for i, bin in enumerate(selected_bins):
            pop[i].X[0] = copy.deepcopy(self.me_archive[bin][0])

        return np.arange(0,n_select*n_parents).reshape((n_select, n_parents))