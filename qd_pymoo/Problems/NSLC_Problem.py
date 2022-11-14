
import autograd.numpy as anp

from qd_pymoo.Evaluators.IEvaluator import IEvaluationFunction
from qd_pymoo.Evaluators.NoveltyEvaluator import NSLCEvaluator
from qd_pymoo.Problems.GenericProblem import GenericProblem


class BaseNSLCProblem(GenericProblem):
    def __init__(self, n_var : int, fitness_evaluator : IEvaluationFunction, nslc_archive : NSLCEvaluator, **kwargs):
        super().__init__(n_var=n_var, n_obj=2, evaluators={'nslc_evaluator' : nslc_archive},  **kwargs)
        self.fitness_evaluator = fitness_evaluator

    def _evaluate(self, x, out, *args, **kwargs):
        f = self.fitness_evaluator.evaluation_fn(x, *args, **kwargs)
        super()._evaluate(x, out, *args, fitness_scores = f, **kwargs)

    def _calc_pareto_front(self):
        return 0

    def _calc_pareto_set(self):
        return anp.full(self.n_var, 0)
