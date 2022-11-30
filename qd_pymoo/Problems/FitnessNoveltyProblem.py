
import autograd.numpy as anp

from qd_pymoo.Evaluators.IEvaluator import IEvaluationFunction
from qd_pymoo.Evaluators.NoveltyEvaluator import NoveltyEvaluatorKD
from qd_pymoo.Problems.GenericProblem import GenericProblem


class BaseFitnessNoveltyProblem(GenericProblem):
    def __init__(self, n_var : int, fitness_evaluator : IEvaluationFunction, novelty_archive : NoveltyEvaluatorKD, **kwargs):
        super().__init__(n_var, n_obj=2, evaluators={'fitness_evaluator' : fitness_evaluator, 'novelty_evaluator' : novelty_archive},  **kwargs)

    def _calc_pareto_front(self):
        return 0

    def _calc_pareto_set(self):
        return anp.full(self.n_var, 0)
