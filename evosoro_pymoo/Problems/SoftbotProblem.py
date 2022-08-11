

import os
import numpy as np
import logging
from typing import List
from abc import ABC, abstractmethod
from pymoo.core.problem import Problem
from common.Utils import readFromPickle, saveToDill, saveToPickle

from evosoro.softbot import SoftBot
from evosoro_pymoo.Evaluators.PhysicsEvaluator import BaseSoftBotPhysicsEvaluator
from evosoro_pymoo.common.ICheckpoint import ICheckpoint
from evosoro_pymoo.common.IRecoverFromFile import IFileRecovery
from evosoro_pymoo.common.IStart import IStarter

logger = logging.getLogger(f"__main__.{__name__}")

class BaseSoftbotProblem(Problem, ABC, ICheckpoint, IStarter):

    def __init__(self, physics_evaluator : BaseSoftBotPhysicsEvaluator, n_var=-1, n_obj=1, n_constr=0):
        super().__init__(n_var, n_obj, n_constr)
        self.evaluators = {"physics" : physics_evaluator}

    def _evaluate(self, x, out, *args, **kwargs):

        X = self._doEvaluations([x_i[0].X for x_i in x], *args, **kwargs)
        objectives_mat = []
        constraints_mat = []
        for x_i in X:
            objectives_mat += [self._extractObjectives(x_i)]
            constraint = self._extractConstraints(x_i)
            constraints_mat += [constraint] if constraint else []
        out["F"] = np.array(objectives_mat, dtype=float)
        if constraints_mat:
            out["G"] = np.array(constraints_mat, dtype=float)
        
        self.backup()
    
    def _doEvaluations(self, X : List[SoftBot], *args, **kwargs):
        for _, evaluator in self.evaluators.items():
            X = evaluator.evaluate(X, *args, **kwargs)
        return X

    def _extractConstraints(self, x : SoftBot) -> List[float]:
        pass
    
    @abstractmethod
    def _extractObjectives(self, x : SoftBot) -> List[float]:
        pass

    def start(self, **kwargs):
        resuming_run = kwargs["resuming_run"]
        for k in self.evaluators.keys():
            if issubclass(type(self.evaluators[k]), IFileRecovery):
                if resuming_run:
                    self.evaluators[k] = self.evaluators[k].file_recovery()
            if issubclass(type(self.evaluators[k]), IStarter):
                self.evaluators[k].start(**kwargs)
    
    def backup(self, *args, **kwargs):
        for _, evaluator in self.evaluators.items():
            if issubclass(type(evaluator), ICheckpoint):
                evaluator.backup(*args, **kwargs)

