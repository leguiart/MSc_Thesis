
import numpy as np
from typing import List

from constants import *
from evosoro.softbot import SoftBot
from evosoro_pymoo.Evaluators.GenotypeDiversityEvaluator import GenotypeDiversityEvaluator
from evosoro_pymoo.Evaluators.IEvaluator import IEvaluator
from evosoro_pymoo.Evaluators.PhysicsEvaluator import BaseSoftBotPhysicsEvaluator

from qd_pymoo.Algorithm.ME_Archive import MAP_ElitesArchive
from qd_pymoo.Problems.FitnessNoveltyProblem import BaseFitnessNoveltyProblem
from qd_pymoo.Problems.NSLC_Problem import BaseNSLCProblem
from qd_pymoo.Evaluators.NoveltyEvaluator import NSLCEvaluator, NoveltyEvaluatorKD
from qd_pymoo.Problems.SingleObjectiveProblem import BaseSingleObjectiveProblem
from qd_pymoo.Problems.ME_Problem import BaseMEProblem

MAX_NS_ARCHIVE_SIZE = (IND_SIZE[0]*IND_SIZE[1]*IND_SIZE[2])**2//2

def unaligned_distance_metric(a, b):
    a_vec = np.array([a.active, a.passive])
    b_vec = np.array([b.active, b.passive])
    return np.sqrt(np.sum((a_vec - b_vec)**2))

def unaligned_vector(a):
    return np.array([a.active, a.passive])

def aligned_distance_metric(a, b):
    a_vec = np.array([a.fitnessX, a.fitnessY])
    b_vec = np.array([b.fitnessX, b.fitnessY])
    return np.sqrt(np.sum((a_vec - b_vec)**2))

def aligned_vector(a):
    return np.array([a.finalX - a.initialX, a.finalY - a.initialY, a.finalZ - a.initialZ])

def is_valid_func(x : SoftBot):
    return x.phenotype.is_valid()


class GenotypeDiversityExtractor(IEvaluator[SoftBot]):

    def __init__(self, genotypeDiversityEvaluator : GenotypeDiversityEvaluator) -> None:
        super().__init__()
        self.genotypeDiversityEvaluator = genotypeDiversityEvaluator
        self.gene_div_matrix = []


    def evaluate(self, X : List[SoftBot], *args, **kwargs) -> List[SoftBot]:
        for indx, individual in enumerate(X):
            individual.control_gene_div = self.genotypeDiversityEvaluator[indx][0]
            individual.morpho_gene_div = self.genotypeDiversityEvaluator[indx][1]
            individual.gene_diversity = self.genotypeDiversityEvaluator[indx][2]
        return X
            

class SoftBotProblemFitness(BaseSingleObjectiveProblem):
    def __init__(self, physics_evaluator: BaseSoftBotPhysicsEvaluator):
        super().__init__(n_var=1, fitness_evaluator = physics_evaluator)

    def _evaluate(self, x, out, *args, **kwargs):
        softBotPop = [vec[0] for vec in x]
        super()._evaluate(softBotPop, out, *args, **kwargs)

class SoftBotProblemFitnessNovelty(BaseFitnessNoveltyProblem):
    def __init__(self, physics_evaluator: BaseSoftBotPhysicsEvaluator, novelty_archive: NoveltyEvaluatorKD):
        super().__init__(n_var=1, fitness_evaluator = physics_evaluator, novelty_archive=novelty_archive)

    def _evaluate(self, x, out, *args, **kwargs):
        softBotPop = [vec[0] for vec in x]
        super()._evaluate(softBotPop, out, *args, **kwargs)
        for i, bot in enumerate(softBotPop):
            bot.unaligned_novelty = -out["F"][i,1]

class SoftBotProblemNSLC(BaseNSLCProblem):
    def __init__(self, physics_evaluator : BaseSoftBotPhysicsEvaluator, nslc_archive : NSLCEvaluator):
        super().__init__(n_var=1, fitness_evaluator=physics_evaluator, nslc_archive=nslc_archive)

    def _evaluate(self, x, out, *args, **kwargs):
        softBotPop = [vec[0] for vec in x]
        super()._evaluate(softBotPop, out, *args, **kwargs)
        for i, bot in enumerate(softBotPop):
            bot.unaligned_novelty = -out["F"][i,1]

class SoftBotProblemME(BaseMEProblem):
    def __init__(self, physics_evaluator : BaseSoftBotPhysicsEvaluator, me_archive: MAP_ElitesArchive):
        super().__init__(n_var=1, fitness_evaluator = physics_evaluator, me_archive = me_archive)

    def _evaluate(self, x, out, *args, **kwargs):
        softBotPop = [vec[0] for vec in x]
        super()._evaluate(softBotPop, out, *args, **kwargs)