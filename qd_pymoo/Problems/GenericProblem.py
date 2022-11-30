

import numpy as np
from typing import List, Dict
from abc import abstractmethod
from pymoo.core.problem import Problem

from qd_pymoo.Evaluators.IEvaluator import IEvaluationFunction


class GenericProblem(Problem):

    def __init__(self, n_var : int, n_obj : int, evaluators : Dict[str, IEvaluationFunction], **kwargs):
        super().__init__(n_var, n_obj,  **kwargs)
        self.evaluators = evaluators

    def _evaluate(self, x, out, *args, **kwargs):

        out["F"] = np.vstack(self._doEvaluations(x, *args, **kwargs)).T
        # if constraints_mat:
        #     out["G"] = np.array(constraints_mat, dtype=float)
            
    def _doEvaluations(self, x, *args, **kwargs):
        scores_list = []
        for _, evaluator in self.evaluators.items():
            scores_list.append(evaluator.evaluation_fn(x, *args, **kwargs))
        return scores_list

