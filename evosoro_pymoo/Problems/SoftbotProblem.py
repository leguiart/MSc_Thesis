

import random
from typing import List
import numpy as np
import subprocess as sub
from functools import partial
import os
import sys
import math
from pymoo.core.problem import Problem
from Evaluators.NoveltyEvaluator import NoveltyEvaluator, NSLCQuality
from evosoro.softbot import SoftBot

# Appending repo's root dir in the python path to enable subsequent imports
from evosoro_pymoo.Evaluators.PhysicsEvaluator import BasePhysicsEvaluator, evaluate_all_pymoo
# sys.path.append(os.getcwd() + "/../..")
from evosoro.tools.logging import PrintLog


class BaseSoftbotProblem(Problem):

    def __init__(self, physics_evaluator : BasePhysicsEvaluator, n_var=-1, n_obj=1, n_constr=0):
        super().__init__(n_var, n_obj, n_constr)
        # # Setting up the simulation object
        # self.sim = sim
        # self.env = env
        # # Setting up the environment object
        # if not isinstance(env, list):
        #     self.env = [env]

        # self.save_vxa_every = save_vxa_every
        # self.directory = directory
        # self.optimizer_name = name
        # self.max_eval_time = max_eval_time
        # self.time_to_try_again = time_to_try_again
        # self.save_lineages = save_lineages
        # self.already_evaluated = {}
        # self.all_evaluated_individuals_ids = []
        # self.num_env_cycles = 0
        # self.print_log = PrintLog()
        # self.curr_env_idx = 0
        # self.objective_dict = objective_dict
        # self.best_fit_so_far = objective_dict[0]["worst_value"]
        # self.n_gen = 1
        self.evaluators = [physics_evaluator]

    def _evaluate(self, x, out, *args, **kwargs):
        # for x_i in x:
        #     x_i[0].X.fitness = self.objective_dict[0]["worst_value"]

        # evaluate_all_pymoo(self.sim, self.env[self.curr_env_idx], x, self.print_log, self.save_vxa_every, self.directory, self.optimizer_name, 
        # self.already_evaluated, self.all_evaluated_individuals_ids, self.objective_dict, self, self.max_eval_time, self.time_to_try_again)
        # X1 = self.physics_evaluator.evaluate([x_i[0].X for x_i in x])
        # X = self.novelty_evaluator.evaluate(X1)
        X = self._doEvaluations([x_i[0].X for x_i in x])
        objectives_mat = []
        constraints_mat = []
        for x_i in X:
            objectives_mat += [self._extractObjectives(x_i)]
            constraint = self._extractConstraints(x_i)
            constraints_mat += [constraint] if constraint else []
        out["F"] = np.array(objectives_mat, dtype=float)
        if constraints_mat:
            out["G"] = np.array(constraints_mat, dtype=float)
    
    def _extractConstraints(self, x : SoftBot) -> List[float]:
        pass

    def _extractObjectives(self, x : SoftBot) -> List[float]:
        raise NotImplementedError

    def _doEvaluations(self, X : List[SoftBot]):
        for evaluator in self.evaluators:
            X = evaluator.evaluate(X)
        return X

def unaligned_distance_metric(a, b):
    a_vec = np.array([a.active, a.passive])
    b_vec = np.array([b.active, b.passive])
    return np.sqrt(np.sum((a_vec - b_vec)**2))

def aligned_distance_metric(a, b):
    a_vec = np.array([a.fitnessX, a.fitnessY])
    b_vec = np.array([b.fitnessX, b.fitnessY])
    return np.sqrt(np.sum((a_vec - b_vec)**2))



class QualitySoftbotProblem(BaseSoftbotProblem):

    #def __init__(self, physics_evaluator : BasePhysicsEvaluator, sim, env, save_vxa_every, directory, name, max_eval_time, time_to_try_again, save_lineages, objective_dict):
    def __init__(self, physics_evaluator : BasePhysicsEvaluator):
        super().__init__(physics_evaluator, n_var=1, n_obj=3, n_constr=0)
        self.evaluators += [NoveltyEvaluator(unaligned_distance_metric, "unaligned_novelty")]

    def _extractObjectives(self, x: SoftBot) -> List[float]:
        return [-x.fitness, x.num_voxels, x.active]


class QualityNoveltySoftbotProblem(BaseSoftbotProblem):

    def __init__(self, physics_evaluator : BasePhysicsEvaluator):
        super().__init__(physics_evaluator, n_var=1, n_obj=3, n_constr=0)
        self.evaluators += [NoveltyEvaluator(unaligned_distance_metric, "unaligned_novelty")]

    def _extractObjectives(self, x: SoftBot) -> List[float]:
        return [-x.fitness, -x.unaligned_novelty, x.active]


class MNSLCSoftbotProblem(BaseSoftbotProblem):

    def __init__(self, physics_evaluator : BasePhysicsEvaluator):
        super().__init__(physics_evaluator, n_var=1, n_obj=4, n_constr=0)
        self.evaluators += [NoveltyEvaluator(aligned_distance_metric, "aligned_novelty"), 
                            NoveltyEvaluator(unaligned_distance_metric, "unaligned_novelty", "unaligned_neighbors"),
                            NSLCQuality()]
        # self.aligned_novelty_evaluator = NoveltyEvaluator(aligned_distance_metric, "aligned_novelty")
        # self.unaligned_novelty_evaluator = NoveltyEvaluator(unaligned_distance_metric, "unaligned_novelty", "unaligned_neighbors")

    def _extractObjectives(self, x: SoftBot) -> List[float]:
        return [-x.nslc_quality, -x.aligned_novelty, -x.unaligned_novelty, x.active]

    # def _evaluate(self, x, out, *args, **kwargs):
    #     X = self.physics_evaluator.evaluate([x_i[0].X for x_i in x])
    #     X1 = self.aligned_novelty_evaluator.evaluate(X)
    #     X2 = self.unaligned_novelty_evaluator.evaluate(X1)
    #     self.compute_nslc_quality(X2)
    #     objectives_li = []
    #     constraints_li = []
    #     for x_i in X2:
    #         objectives_li += [[-x_i.nslc_quality, -x_i.aligned_novelty, -x_i.unaligned_novelty, x_i.active]]
    #         # constraints_li += [[-x_i.passive/x_i.active + 0.2]]
    #     out["F"] = np.array(objectives_li, dtype=float)
    #     # out["G"] = np.array(constraints_li, dtype=float)
