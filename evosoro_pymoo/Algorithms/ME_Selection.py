
import random
import numpy as np
import copy

from pymoo.core.selection import Selection
from evosoro_pymoo.Algorithms.MAP_Elites import MAP_ElitesArchive


class MESelection(Selection):

    def __init__(self, me_archive : MAP_ElitesArchive, fitness_attr : str = 'fitness') -> None:
        self.me_archive = me_archive
        self.fitness_attr = fitness_attr

    def _do(self, pop, n_select, n_parents=1, **kwargs):

        # Get indexes of filled bins
        selected_bins = []
        filled_bins_indxs = self.me_archive.filled_indices
        if len(filled_bins_indxs) < n_select * n_parents:
            individuals = []
            fitness_scores = []

            for indx in filled_bins_indxs:
                individuals += [self.me_archive[indx]]

            individuals.sort(key=lambda x : getattr(x, self.fitness_attr))
            fitness_scores = list(map(lambda x : getattr(x, self.fitness_attr), individuals))
            normalized_fitness_scores = np.array(fitness_scores)/sum(fitness_scores)
            probabilities = np.cumsum(normalized_fitness_scores)

            num_missing_indices = n_select * n_parents - len(filled_bins_indxs)
            chosen_array_indices = np.searchsorted(probabilities, np.random.rand(num_missing_indices))
            chosen_archive_indices = list(map(lambda x : self.me_archive.feature_descriptor_idx(individuals[x]), chosen_array_indices))
            selected_bins += chosen_archive_indices
                

        # Uniformly select filled bins
        selected_bins += random.choices(filled_bins_indxs, k = min(n_select*n_parents, len(filled_bins_indxs)))
        for i, bin in enumerate(selected_bins):
            pop[i].X = copy.deepcopy(self.me_archive[bin])

        return np.arange(0,n_select*n_parents).reshape((n_select, n_parents))