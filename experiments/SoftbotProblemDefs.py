

import numpy as np
from typing import List

from common.Constants import *
from evosoro.softbot import SoftBot
from evosoro_pymoo.Evaluators.PhysicsEvaluator import BasePhysicsEvaluator
from evosoro_pymoo.Evaluators.NoveltyEvaluator import NoveltyEvaluator, NSLCQuality
from evosoro_pymoo.Problems.SoftbotProblem import BaseSoftbotProblem

MAX_NS_ARCHIVE_SIZE = (IND_SIZE[0]*IND_SIZE[1]*IND_SIZE[2])**2//2

def unaligned_distance_metric(a, b):
    a_vec = np.array([a.active, a.passive])
    b_vec = np.array([b.active, b.passive])
    return np.sqrt(np.sum((a_vec - b_vec)**2))

def aligned_distance_metric(a, b):
    a_vec = np.array([a.fitnessX, a.fitnessY])
    b_vec = np.array([b.fitnessX, b.fitnessY])
    return np.sqrt(np.sum((a_vec - b_vec)**2))

def aligned_distance_metric_gpu(a, b):
    a_vec = np.array([a.finalX - a.initialX, a.finalY - a.initialY])
    b_vec = np.array([b.finalX - b.initialX, b.finalY - b.initialY])
    return np.sqrt(np.sum((a_vec - b_vec)**2))

def is_valid_func(x : SoftBot):
    return x.phenotype.is_valid()

class QualitySoftbotProblem(BaseSoftbotProblem):

    def __init__(self, physics_evaluator : BasePhysicsEvaluator, pop_size):
        super().__init__(physics_evaluator, n_var=1, n_obj=1)
        self.evaluators["unaligned_novelty"] = NoveltyEvaluator(unaligned_distance_metric, "unaligned_novelty", is_valid_func=is_valid_func, min_novelty_archive_size=pop_size//2, max_novelty_archive_size=int(MAX_NS_ARCHIVE_SIZE*0.68), k_neighbors=20)

    def _extractObjectives(self, x: SoftBot) -> List[float]:
        return [-x.fitness]

    # def _extractConstraints(self, x: SoftBot) -> List[float]:
    #     return [-x.num_voxels + 0.1*x.genotype.ds_size, x.num_voxels - 0.9*x.genotype.ds_size, -x.active + 0.1*x.num_voxels, x.active - 0.8*x.num_voxels]


class QualityNoveltySoftbotProblem(BaseSoftbotProblem):

    def __init__(self, physics_evaluator : BasePhysicsEvaluator, pop_size):
        super().__init__(physics_evaluator, n_var=1, n_obj=2)
        self.evaluators["unaligned_novelty"] = NoveltyEvaluator(unaligned_distance_metric, "unaligned_novelty", is_valid_func=is_valid_func, min_novelty_archive_size=pop_size, max_novelty_archive_size=int(MAX_NS_ARCHIVE_SIZE*0.68), k_neighbors=20)

    def _extractObjectives(self, x: SoftBot) -> List[float]:
        return [-x.fitness, -x.unaligned_novelty]
    
    # def _extractConstraints(self, x: SoftBot) -> List[float]:
    #     return [-x.num_voxels + 0.1*x.genotype.ds_size, x.num_voxels - 0.9*x.genotype.ds_size, -x.active + 0.1*x.num_voxels, x.active - 0.8*x.num_voxels]

class NSLCSoftbotProblem(BaseSoftbotProblem):

    def __init__(self, physics_evaluator : BasePhysicsEvaluator, pop_size):
        super().__init__(physics_evaluator, n_var=1, n_obj=2)
        self.evaluators.update({"unaligned_novelty" : NoveltyEvaluator(unaligned_distance_metric, "unaligned_novelty", nslc_neighbors_name="unaligned_neighbors", is_valid_func=is_valid_func, min_novelty_archive_size=pop_size, max_novelty_archive_size=int(MAX_NS_ARCHIVE_SIZE*0.68), k_neighbors=20),
                            "nslc_quality" : NSLCQuality()})

    def _extractObjectives(self, x: SoftBot) -> List[float]:
        return [-x.nslc_quality, -x.unaligned_novelty]

    # def _extractConstraints(self, x: SoftBot) -> List[float]:
    #     return [-x.num_voxels + 0.1*x.genotype.ds_size, x.num_voxels - 0.9*x.genotype.ds_size, -x.active + 0.1*x.num_voxels, x.active - 0.8*x.num_voxels]

class MNSLCSoftbotProblem(BaseSoftbotProblem):

    def __init__(self, physics_evaluator : BasePhysicsEvaluator, pop_size):
        super().__init__(physics_evaluator, n_var=1, n_obj=3)
        self.evaluators.update({"aligned_novelty" : NoveltyEvaluator(aligned_distance_metric, "aligned_novelty", is_valid_func=is_valid_func, min_novelty_archive_size=pop_size, max_novelty_archive_size=int(MAX_NS_ARCHIVE_SIZE*0.68), k_neighbors=20), 
                            "unaligned_novelty" : NoveltyEvaluator(unaligned_distance_metric, "unaligned_novelty", nslc_neighbors_name="unaligned_neighbors", is_valid_func=is_valid_func, min_novelty_archive_size=pop_size, max_novelty_archive_size=int(MAX_NS_ARCHIVE_SIZE*0.68), k_neighbors=20),
                            "nslc_quality" : NSLCQuality()})

    def _extractObjectives(self, x: SoftBot) -> List[float]:
        return [-x.nslc_quality, -x.aligned_novelty, -x.unaligned_novelty]

    # def _extractConstraints(self, x: SoftBot) -> List[float]:
    #     return [-x.num_voxels + 0.1*x.genotype.ds_size, x.num_voxels - 0.9*x.genotype.ds_size, -x.active + 0.1*x.num_voxels, x.active - 0.8*x.num_voxels]

class MNSLCSoftbotProblemGPU(BaseSoftbotProblem):

    def __init__(self, physics_evaluator : BasePhysicsEvaluator, pop_size):
        super().__init__(physics_evaluator, n_var=1, n_obj=3)
        self.evaluators.update({"aligned_novelty" : NoveltyEvaluator(aligned_distance_metric_gpu, "aligned_novelty", is_valid_func=is_valid_func, min_novelty_archive_size=pop_size, max_novelty_archive_size=int(MAX_NS_ARCHIVE_SIZE*0.68), k_neighbors=20), 
                            "unaligned_novelty" : NoveltyEvaluator(unaligned_distance_metric, "unaligned_novelty", nslc_neighbors_name="unaligned_neighbors", is_valid_func=is_valid_func, min_novelty_archive_size=pop_size, max_novelty_archive_size=int(MAX_NS_ARCHIVE_SIZE*0.68), k_neighbors=20),
                            "nslc_quality" : NSLCQuality()})

    def _extractObjectives(self, x: SoftBot) -> List[float]:
        return [-x.nslc_quality, -x.aligned_novelty, -x.unaligned_novelty]

    # def _extractConstraints(self, x: SoftBot) -> List[float]:
    #     return [-x.num_voxels + 0.1*x.genotype.ds_size, x.num_voxels - 0.9*x.genotype.ds_size, -x.active + 0.1*x.num_voxels, x.active - 0.9*x.num_voxels]