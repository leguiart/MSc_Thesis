
import autograd.numpy as anp
from pymoo.core.problem import Problem
from qd_pymoo.Algorithm.ME_Archive import MAP_ElitesArchive
from qd_pymoo.Evaluators.IEvaluator import IEvaluationFunction

from qd_pymoo.Evaluators.NoveltyEvaluator import NSLCEvaluator, NoveltyEvaluatorKD
from qd_pymoo.Problems.FitnessNoveltyProblem import BaseFitnessNoveltyProblem
from qd_pymoo.Problems.NSLC_Problem import BaseNSLCProblem
from qd_pymoo.Problems.ME_Problem import BaseMEProblem

class AckleyFunction(IEvaluationFunction):
    def __init__(self, n_var = 2, a=20, b=1/5, c=2 * anp.pi):
        self.n_var = n_var
        self.a = a
        self.b = b
        self.c = c
        
    def evaluation_fn(self, X, *args, **kwargs):
        part1 = -1. * self.a * anp.exp(-1. * self.b * anp.sqrt((1. / self.n_var) * anp.sum(X * X, axis=1)))
        part2 = -1. * anp.exp((1. / self.n_var) * anp.sum(anp.cos(self.c * X), axis=1))
        f = part1 + part2 + self.a + anp.exp(1)
        return f

class RastriginFunction(IEvaluationFunction):
    def __init__(self, n_var=2, A=10.0):
        self.A = A
        self.n_var = n_var

    def evaluation_fn(self, X, *args, **kwargs):
        z = anp.power(X, 2) - self.A * anp.cos(2 * anp.pi * X)
        f = self.A * self.n_var + anp.sum(z, axis=1)
        return f

class Schaffer4Function(IEvaluationFunction):
    def evaluation_fn(self, X, *args, **kwargs):
        f = .5 + (anp.cos(anp.sin(anp.abs(X[:, 0]**2 - X[:, 1]**2)))**2 - .5)/(1. + 0.001*(X[:, 0]**2 + X[:, 1]**2))**2
        return f

class EggholderFunction(IEvaluationFunction):
    def evaluation_fn(self, X, *args, **kwargs):
        f = - (X[:, 1] + 47.)*anp.sin(anp.sqrt(anp.abs(X[:,0]/2 + X[:,1] + 47.))) - X[:,0]*anp.sin(anp.sqrt(anp.abs(X[:,0] - (X[:,1] + 47.))))
        return f


class SingleObjectiveProblem(Problem):
    def __init__(self, n_var, evaluator : IEvaluationFunction, *args, **kwargs):
        super().__init__(n_var, *args, **kwargs)
        self.evaluator = evaluator

    def _evaluate(self, x, out):
        out["F"] = self.evaluator.evaluation_fn(x)
    
    def _calc_pareto_front(self):
        return 0

    def _calc_pareto_set(self):
        return anp.full(self.n_var, 0)


class AckleyProblem(SingleObjectiveProblem):
    def __init__(self, n_var=2, a=20, b=1/5, c=2 * anp.pi):
        super().__init__(n_var, AckleyFunction(n_var, a, b, c), n_constr=0, xl=-32.768, xu=+32.768, type_var=anp.double)

class RastriginProblem(SingleObjectiveProblem):
    def __init__(self, n_var=2, A=10.0):
        super().__init__(n_var, RastriginFunction(n_var, A), n_constr=0, xl=-5, xu=5, type_var=anp.double)

class Schaffer4Problem(SingleObjectiveProblem):
    def __init__(self, n_var = 2):
        super().__init__(n_var, Schaffer4Function(), n_constr=0, xl=-100, xu=100, type_var=anp.double)

class EggholderProblem(SingleObjectiveProblem):
    def __init__(self, n_var = 2):
        super().__init__(n_var, EggholderFunction(), n_constr=0, xl=-512, xu=512, type_var=anp.double)


class NSLC_Ackley(BaseNSLCProblem):
    def __init__(self, n_var=2, a=20, b=1/5, c=2 * anp.pi, nslc_archive = NSLCEvaluator("nslc_ackley")):
        super().__init__(n_var, AckleyFunction(n_var, a, b, c), nslc_archive, n_constr=0, xl=-32.768, xu=+32.768, type_var=anp.double)
        self.a = a
        self.b = b
        self.c = c

class NSLC_Rastrigin(BaseNSLCProblem):
    def __init__(self, n_var=2, A=10.0, nslc_archive = NSLCEvaluator("nslc_rastrigin")):
        super().__init__(n_var, RastriginFunction(n_var, A), nslc_archive, n_constr=0, xl=-5, xu=5, type_var=anp.double)
        self.A = A

class NSLC_Schaffer4(BaseNSLCProblem):
    def __init__(self, n_var=2, nslc_archive = NSLCEvaluator("nslc_schaffer4")):
        super().__init__(n_var, Schaffer4Function(), nslc_archive, n_constr=0, xl=-100, xu=100, type_var=anp.double)
        
class NSLC_Eggholder(BaseNSLCProblem):
    def __init__(self, n_var=2, nslc_archive = NSLCEvaluator("nslc_eggholder")):
        super().__init__(n_var, EggholderFunction(), nslc_archive, n_constr=0, xl=-512, xu=512, type_var=anp.double)

class FN_Ackley(BaseFitnessNoveltyProblem):
    def __init__(self, n_var=2, a=20, b=1/5, c=2 * anp.pi, novelty_archive = NoveltyEvaluatorKD("nov_ackley")):
        super().__init__(n_var, AckleyFunction(n_var, a, b, c), novelty_archive, n_constr=0, xl=-32.768, xu=+32.768, type_var=anp.double)
        self.a = a
        self.b = b
        self.c = c

class FN_Rastrigin(BaseFitnessNoveltyProblem):
    def __init__(self, n_var=2, A=10.0, novelty_archive = NoveltyEvaluatorKD("nov_rastrigin")):
        super().__init__(n_var, RastriginFunction(n_var, A), novelty_archive, n_constr=0, xl=-5, xu=5, type_var=anp.double)
        self.A = A

class FN_Schaffer4(BaseFitnessNoveltyProblem):
    def __init__(self, n_var=2, novelty_archive = NoveltyEvaluatorKD("nov_schaffer4")):
        super().__init__(n_var, Schaffer4Function(), novelty_archive, n_constr=0, xl=-100, xu=100, type_var=anp.double)
        
class FN_Eggholder(BaseFitnessNoveltyProblem):
    def __init__(self, n_var=2, novelty_archive = NoveltyEvaluatorKD("nov_eggholder")):
        super().__init__(n_var, EggholderFunction(), novelty_archive, n_constr=0, xl=-512, xu=512, type_var=anp.double)


class ME_Ackley(BaseMEProblem):
    def __init__(self, me_archive : MAP_ElitesArchive, n_var=2, a=20, b=1/5, c=2 * anp.pi):
        super().__init__(n_var, AckleyFunction(n_var, a, b, c), me_archive, n_constr=0, xl=-32.768, xu=+32.768, type_var=anp.double)
        self.a = a
        self.b = b
        self.c = c

class ME_Rastrigin(BaseMEProblem):
    def __init__(self, me_archive : MAP_ElitesArchive, n_var=2, A=10.0):
        super().__init__(n_var, RastriginFunction(n_var, A), me_archive, n_constr=0, xl=-5, xu=5, type_var=anp.double)
        self.A = A

class ME_Schaffer4(BaseMEProblem):
    def __init__(self, me_archive : MAP_ElitesArchive, n_var=2):
        super().__init__(n_var, Schaffer4Function(), me_archive, n_constr=0, xl=-100, xu=100, type_var=anp.double)

class ME_Eggholder(BaseMEProblem):
    def __init__(self, me_archive : MAP_ElitesArchive, n_var=2):
        super().__init__(n_var, EggholderFunction(), me_archive, n_constr=0, xl=-512, xu=512, type_var=anp.double)