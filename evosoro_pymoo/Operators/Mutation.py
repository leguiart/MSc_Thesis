
import copy
import logging
import random
import os
import sys
import inspect
import numpy as np

from pymoo.core.mutation import Mutation


logger = logging.getLogger(f"__main__.{__name__}")


class SoftbotMutation(Mutation):
    def __init__(self, max_id):
        self.max_id = max_id
        super().__init__()

    def _do(self, problem, X, **kwargs):
        self.max_id = len(X) if self.max_id == 0 else self.max_id
        return self.create_new_children_through_mutation(X, problem.evaluators["physics"].objective_dict)


    def create_new_children_through_mutation(self, pop, objective_dict, new_children=None, 
                                            mutate_network_probs=None, max_mutation_attempts=1500):
        """Create copies, with modification, of existing individuals in the population.

        Parameters
        ----------
        pop : Population class
            This provides the individuals to mutate.

        print_log : PrintLog()
            For logging

        new_children : a list of new children created outside this function (may be empty)
            This is useful if creating new children through multiple functions, e.g. Crossover and Mutation.

        mutate_network_probs : probability, float between 0 and 1 (inclusive)
            The probability of mutating each network.

        max_mutation_attempts : int
            Maximum number of invalid mutation attempts to allow before giving up on mutating a particular individual.

        Returns
        -------
        new_children : list
            A list of new individual SoftBots.

        """
        if new_children is None:
            new_children = []

        random.shuffle(pop)

        while len(new_children) < len(pop):
            for ind in pop:
                new_individual = mutate_individual(ind, mutate_network_probs, objective_dict, max_mutation_attempts)
                new_individual.id = self.max_id
                self.max_id += 1
                new_children.append(new_individual)

        return new_children

class ME_SoftbotMutation(Mutation):
    def __init__(self, max_id, offspring_n):
        self.max_id = max_id
        self.offspring_n = offspring_n
        super().__init__()


    def _do(self, problem, X, **kwargs):
        self.max_id = len(X) if self.max_id == 0 else self.max_id
        return self.mutate_from_archive(problem.evaluators["map_elites_archive_f"], problem.evaluators["physics"].objective_dict)


    def mutate_from_archive(self, archive, objective_dict, new_children=None, 
                            mutate_network_probs=None, max_mutation_attempts=1500):
        """Create copies, with modification, of existing individuals in the population.

        Parameters
        ----------
        pop : Population class
            This provides the individuals to mutate.

        print_log : PrintLog()
            For logging

        new_children : a list of new children created outside this function (may be empty)
            This is useful if creating new children through multiple functions, e.g. Crossover and Mutation.

        mutate_network_probs : probability, float between 0 and 1 (inclusive)
            The probability of mutating each network.

        max_mutation_attempts : int
            Maximum number of invalid mutation attempts to allow before giving up on mutating a particular individual.

        Returns
        -------
        new_children : list
            A list of new individual SoftBots.

        """
        if new_children is None:
            new_children = []

        # Get indexes of filled bins
        filled_bins_indxs = [indx for indx, val in enumerate(archive.filled_elites_archive) if val == 1]

        # Uniformly select individuals to mutate
        bins_to_mutate = random.sample(filled_bins_indxs, self.offspring_n)

        while len(new_children) < self.offspring_n:
            for indx in bins_to_mutate:
                ind = archive[indx]
                new_individual = mutate_individual(ind, mutate_network_probs, objective_dict, max_mutation_attempts)
                new_individual.id = self.max_id
                self.max_id += 1
                new_children.append(new_individual)

        return new_children

def mutate_individual(ind, mutate_network_probs, objective_dict, max_mutation_attempts):
    clone = copy.deepcopy(ind)
    
    if mutate_network_probs is None:
        required = 0
    else:
        required = mutate_network_probs.count(1)

    selection = []
    while np.sum(selection) <= required:
        if mutate_network_probs is None:
            # uniformly select networks
            selection = np.random.random(len(clone.genotype)) < 1 / float(len(clone.genotype))
        else:
            # use probability distribution
            selection = np.random.random(len(clone.genotype)) < mutate_network_probs

        # don't select any frozen networks (used to freeze aspects of genotype during evolution)
        for idx in range(len(selection)):
            if clone.genotype[idx].freeze:
                selection[idx] = False

    selected_networks = np.arange(len(clone.genotype))[selection].tolist()

    for rank, goal in objective_dict.items():
        setattr(clone, "parent_{}".format(goal["name"]), getattr(clone, goal["name"]))

    clone.parent_genotype = ind.genotype
    clone.parent_id = clone.id

    for name, details in clone.genotype.to_phenotype_mapping.items():
        details["old_state"] = copy.deepcopy(details["state"])

    # old_individual = copy.deepcopy(clone)

    for selected_net_idx in selected_networks:
        mutation_counter = 0
        done = False
        while not done:
            mutation_counter += 1
            candidate = copy.deepcopy(clone)

            # perform mutation(s)
            for _ in range(candidate.genotype[selected_net_idx].num_consecutive_mutations):
                if not clone.genotype[selected_net_idx].direct_encoding:
                    # using CPPNs
                    mut_func_args = inspect.getargspec(candidate.genotype[selected_net_idx].mutate)
                    mut_func_args = [0 for _ in range(1, len(mut_func_args.args))]
                    choice = random.choice(range(len(mut_func_args)))
                    mut_func_args[choice] = 1
                    variation_type, variation_degree = candidate.genotype[selected_net_idx].mutate(*mut_func_args)
                else:
                    # direct encoding with possibility of evolving mutation rate
                    # TODO: enable cppn mutation rate evolution
                    rate = None
                    for net in clone.genotype:
                        if "mutation_rate" in net.output_node_names:
                            rate = net.values  # evolved mutation rates, one for each voxel
                    if "mutation_rate" not in candidate.genotype[selected_net_idx].output_node_names:
                        # use evolved mutation rates
                        variation_type, variation_degree = candidate.genotype[selected_net_idx].mutate(rate)
                    else:
                        # this is the mutation rate itself (use predefined meta-mutation rate)
                        variation_type, variation_degree = candidate.genotype[selected_net_idx].mutate()

            if variation_degree != "":
                candidate.variation_type = "{0}({1})".format(variation_type, variation_degree)
            else:
                candidate.variation_type = str(variation_type)
            candidate.genotype.express()

            if candidate.genotype[selected_net_idx].allow_neutral_mutations:
                done = True
                clone = copy.deepcopy(candidate)  # SAM: ensures change is made to every net
                break
            else:
                for name, details in candidate.genotype.to_phenotype_mapping.items():
                    new = details["state"]
                    old = details["old_state"]
                    changes = np.array(new != old, dtype=np.bool)
                    if np.any(changes) and candidate.phenotype.is_valid():
                        done = True
                        clone = copy.deepcopy(candidate)  # SAM: ensures change is made to every net
                        break
                # for name, details in candidate.genotype.to_phenotype_mapping.items():
                #     if np.sum( details["old_state"] != details["state"] ) and candidate.phenotype.is_valid():
                #         done = True
                #         break

            if mutation_counter > max_mutation_attempts:
                logger.info("Couldn't find a successful mutation in {} attempts! "
                                    "Skipping this network.".format(max_mutation_attempts))
                num_edges = len(clone.genotype[selected_net_idx].graph.edges())
                num_nodes = len(clone.genotype[selected_net_idx].graph.nodes())
                logger.info("num edges: {0}; num nodes {1}".format(num_edges, num_nodes))
                break

        # end while

        if not clone.genotype[selected_net_idx].direct_encoding:
            for output_node in clone.genotype[selected_net_idx].output_node_names:
                clone.genotype[selected_net_idx].graph.nodes[output_node]["old_state"] = ""

    # reset all objectives we calculate in VoxCad to unevaluated values
    for rank, goal in objective_dict.items():
        if goal["tag"] is not None:
            setattr(clone, goal["name"], goal["worst_value"])
    
    return clone
