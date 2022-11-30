
import autograd.numpy as anp
from qd_pymoo.Evaluators.IEvaluator import IEvaluationFunction
from qd_pymoo.Problems.GenericProblem import GenericProblem

class BaseSingleObjectiveProblem(GenericProblem):
    def __init__(self, n_var, fitness_evaluator : IEvaluationFunction, *args, **kwargs):
        super().__init__(n_var=n_var, n_obj=1, evaluators={'fitness_evaluator' : fitness_evaluator},  **kwargs)
    
    def _calc_pareto_front(self):
        return 0

    def _calc_pareto_set(self):
        return anp.full(self.n_var, 0)