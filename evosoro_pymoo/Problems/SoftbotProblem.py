

import random
import numpy as np
import subprocess as sub
from functools import partial
import os
import sys
import math
from pymoo.core.problem import Problem
from Evaluators.NoveltyEvaluator import NoveltyEvaluator

# Appending repo's root dir in the python path to enable subsequent imports
from Evaluators.SoftbotEvaluator import evaluate_all_pymoo
# sys.path.append(os.getcwd() + "/../..")
from evosoro.tools.logging import PrintLog


class BaseSoftbotProblem(Problem):
    def update_env(self):
        if self.num_env_cycles > 0:
            switch_every = self.max_gens / float(self.num_env_cycles)
            self.curr_env_idx = int(self.n_gen / switch_every % len(self.env))
            print (" Using environment {0} of {1}".format(self.curr_env_idx+1, len(self.env)))

    def evaluate(self, X, *args, return_values_of=None, return_as_dictionary=False, **kwargs):
        return super().evaluate(X, *args, return_values_of=return_values_of, return_as_dictionary = return_as_dictionary, **kwargs)


class QualitySoftbotProblem(BaseSoftbotProblem):

    def __init__(self, sim, env, save_vxa_every, directory, name, max_eval_time, time_to_try_again, save_lineages, objective_dict):
        super().__init__(n_var=1, n_obj=3, n_constr=0)
        # Setting up the simulation object
        self.sim = sim
        self.env = env
        # Setting up the environment object
        if not isinstance(env, list):
            self.env = [env]

        self.save_vxa_every = save_vxa_every
        self.directory = directory
        self.optimizer_name = name
        self.max_eval_time = max_eval_time
        self.time_to_try_again = time_to_try_again
        self.save_lineages = save_lineages
        self.already_evaluated = {}
        self.all_evaluated_individuals_ids = []
        self.num_env_cycles = 0
        self.print_log = PrintLog()
        self.curr_env_idx = 0
        self.objective_dict = objective_dict
        self.best_fit_so_far = objective_dict[0]["worst_value"]
        self.n_gen = 1
        self.novelty_evaluator = NoveltyEvaluator(unaligned_distance_metric)

    def _evaluate(self, x, out, *args, **kwargs):
        for x_i in x:
            x_i[0].X.fitness = self.objective_dict[0]["worst_value"]

        evaluate_all_pymoo(self.sim, self.env[self.curr_env_idx], x, self.print_log, self.save_vxa_every, self.directory, self.optimizer_name, 
        self.already_evaluated, self.all_evaluated_individuals_ids, self.objective_dict, self, self.max_eval_time, self.time_to_try_again)
        X = self.novelty_evaluator.evaluate([x_i[0].X for x_i in x], "unaligned_novelty")

        fitness_li = []
        for x_i in X:
            fitness_li += [[-x_i.fitness, x_i.num_voxels, x_i.active]]
        out["F"] = np.array(fitness_li, dtype=float)


def unaligned_distance_metric(a, b):
    a_vec = np.array([a.active, a.passive])
    b_vec = np.array([b.active, b.passive])
    return np.sqrt(np.sum((a_vec - b_vec)**2))

class QualityNoveltySoftbotProblem(BaseSoftbotProblem):

    def __init__(self, sim, env, save_vxa_every, directory, name, max_eval_time, time_to_try_again, save_lineages, objective_dict):
        super().__init__(n_var=1, n_obj=3, n_constr=0)
        # Setting up the simulation object
        self.sim = sim
        self.env = env
        # Setting up the environment object
        if not isinstance(env, list):
            self.env = [env]

        self.save_vxa_every = save_vxa_every
        self.directory = directory
        self.optimizer_name = name
        self.max_eval_time = max_eval_time
        self.time_to_try_again = time_to_try_again
        self.save_lineages = save_lineages
        self.already_evaluated = {}
        self.all_evaluated_individuals_ids = []
        self.num_env_cycles = 0
        self.print_log = PrintLog()
        self.curr_env_idx = 0
        self.objective_dict = objective_dict
        self.best_fit_so_far = objective_dict[0]["worst_value"]
        self.n_gen = 1
        self.novelty_evaluator = NoveltyEvaluator(unaligned_distance_metric)


    def _evaluate(self, x, out, *args, **kwargs):
        for x_i in x:
            x_i[0].X.fitness = self.objective_dict[0]["worst_value"]
        evaluate_all_pymoo(self.sim, self.env[self.curr_env_idx], x, self.print_log, self.save_vxa_every, self.directory, self.optimizer_name, 
        self.already_evaluated, self.all_evaluated_individuals_ids, self.objective_dict, self, self.max_eval_time, self.time_to_try_again)
        X = self.novelty_evaluator.evaluate([x_i[0].X for x_i in x], "unaligned_novelty")

        objectives_li = []
        for x_i in X:
            objectives_li += [[-x_i.fitness, -x_i.unaligned_novelty, x_i.active]]
        out["F"] = np.array(objectives_li, dtype=float)

def aligned_distance_metric(a, b):
    a_vec = np.array([a.fitnessX, a.fitnessY])
    b_vec = np.array([b.fitnessX, b.fitnessY])
    return np.sqrt(np.sum((a_vec - b_vec)**2))


class MNSLCSoftbotProblem(BaseSoftbotProblem):

    def __init__(self, sim, env, save_vxa_every, directory, name, max_eval_time, time_to_try_again, save_lineages, objective_dict):
        super().__init__(n_var=1, n_obj=4, n_constr=0)
        # Setting up the simulation object
        self.sim = sim
        self.env = env
        # Setting up the environment object
        if not isinstance(env, list):
            self.env = [env]

        self.save_vxa_every = save_vxa_every
        self.directory = directory
        self.optimizer_name = name
        self.max_eval_time = max_eval_time
        self.time_to_try_again = time_to_try_again
        self.save_lineages = save_lineages
        self.already_evaluated = {}
        self.all_evaluated_individuals_ids = []
        self.num_env_cycles = 0
        self.print_log = PrintLog()
        self.curr_env_idx = 0
        self.objective_dict = objective_dict
        self.best_fit_so_far = objective_dict[0]["worst_value"]
        self.n_gen = 1
        self.aligned_novelty_evaluator = NoveltyEvaluator(aligned_distance_metric)
        self.unaligned_novelty_evaluator = NoveltyEvaluator(unaligned_distance_metric)

    def compute_nslc_quality(self, X):
        for x in X:
            x.nslc_quality = sum([1 if x.fitness > n else 0 for n in x.unaligned_neighbors])/len(x.unaligned_neighbors)

    def _evaluate(self, x, out, *args, **kwargs):
        for x_i in x:
            x_i[0].X.fitness = self.objective_dict[0]["worst_value"]
        evaluate_all_pymoo(self.sim, self.env[self.curr_env_idx], x, self.print_log, self.save_vxa_every, self.directory, self.optimizer_name, 
        self.already_evaluated, self.all_evaluated_individuals_ids, self.objective_dict, self, self.max_eval_time, self.time_to_try_again)
        X1 = self.aligned_novelty_evaluator.evaluate([x_i[0].X for x_i in x], "aligned_novelty")
        X2 = self.unaligned_novelty_evaluator.evaluate(X1, "unaligned_novelty", "unaligned_neighbors")
        self.compute_nslc_quality(X2)
        objectives_li = []
        constraints_li = []
        for x_i in X2:
            objectives_li += [[-x_i.nslc_quality, -x_i.aligned_novelty, -x_i.unaligned_novelty, x_i.active]]
            # constraints_li += [[-x_i.passive/x_i.active + 0.2]]
        out["F"] = np.array(objectives_li, dtype=float)
        # out["G"] = np.array(constraints_li, dtype=float)
