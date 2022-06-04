

import numpy as np
import logging
from typing import List
from abc import ABC, abstractmethod
from pymoo.core.problem import Problem

from evosoro.softbot import SoftBot
from evosoro_pymoo.Evaluators.PhysicsEvaluator import BasePhysicsEvaluator

logger = logging.getLogger(f"__main__.{__name__}")

class BaseSoftbotProblem(Problem, ABC):

    def __init__(self, physics_evaluator : BasePhysicsEvaluator, n_var=-1, n_obj=1, n_constr=0):
        super().__init__(n_var, n_obj, n_constr)
        self.evaluators = {"physics" : physics_evaluator}

    def _evaluate(self, x, out, *args, **kwargs):

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
    
    @abstractmethod
    def _extractConstraints(self, x : SoftBot) -> List[float]:
        pass
    
    @abstractmethod
    def _extractObjectives(self, x : SoftBot) -> List[float]:
        pass

    def _doEvaluations(self, X : List[SoftBot]):
        for _, evaluator in self.evaluators.items():
            X = evaluator.evaluate(X)
        return X
