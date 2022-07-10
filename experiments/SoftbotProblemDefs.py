

import shutil
import subprocess
import numpy as np
from typing import List

from common.Constants import *
from evosoro.softbot import SoftBot
from evosoro_pymoo.Evaluators.GenotypeDistanceEvaluator import GenotypeDistanceEvaluator
from evosoro_pymoo.Evaluators.GenotypeDiversityEvaluator import GenotypeDiversityEvaluator
from evosoro_pymoo.Evaluators.IEvaluator import EvaluatorInterface, IEvaluator
from evosoro_pymoo.Evaluators.PhysicsEvaluator import BaseSoftBotPhysicsEvaluator
from evosoro_pymoo.Evaluators.NoveltyEvaluator import NSLCEvaluator, NoveltyEvaluatorKD
from evosoro_pymoo.Problems.SoftbotProblem import BaseSoftbotProblem

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
    return np.array([a.fitnessX, a.fitnessY])

def aligned_distance_metric_gpu(a, b):
    a_vec = np.array([a.finalX - a.initialX, a.finalY - a.initialY])
    b_vec = np.array([b.finalX - b.initialX, b.finalY - b.initialY])
    return np.sqrt(np.sum((a_vec - b_vec)**2))

def aligned_vector_gpu(a):
    return np.array([a.finalX, a.finalY])

def is_valid_func(x : SoftBot):
    return x.phenotype.is_valid()


class GenotypeDiversityExtractor(IEvaluator[SoftBot]):

    def __init__(self, genotypeDiversityEvaluator : GenotypeDiversityEvaluator) -> None:
        super().__init__()
        self.genotypeDiversityEvaluator = genotypeDiversityEvaluator
        self.gene_div_matrix = []


    def evaluate(self, X : List[SoftBot], *args, **kwargs) -> List[SoftBot]:
        for indx, individual in enumerate(X):
            individual.gene_diversity = np.mean(self.genotypeDiversityEvaluator[indx])
            individual.control_gene_div = self.genotypeDiversityEvaluator[indx][0]
            individual.morpho_gene_div = self.genotypeDiversityEvaluator[indx][1]
        return X
            

class PopulationSaver(IEvaluator[SoftBot]):

    def __init__(self, backup_path : str) -> None:
        self.backup_path =backup_path


    def evaluate(self, X : List[SoftBot], *args, **kwargs) -> List[SoftBot]:
        subprocess.call("rm -rf " + self.backup_path + "/pickledPops" + "/* 2>/dev/null", shell=True)
        for individual in X:
            individual.save_softbot_backup(self.backup_path + "/pickledPops")
        return X


class QualitySoftbotProblem(BaseSoftbotProblem):

    def __init__(self, physics_evaluator : BaseSoftBotPhysicsEvaluator, pop_size : int, backup_path : str):
        super().__init__(physics_evaluator, n_var=1, n_obj=1)
        genotypeDiversityEvaluator = GenotypeDiversityEvaluator()
        populationSaver = PopulationSaver(backup_path)
        # self.evaluators["unaligned_novelty"] = NoveltyEvaluator(unaligned_distance_metric, "unaligned_novelty", is_valid_func=is_valid_func, min_novelty_archive_size=pop_size//2, max_novelty_archive_size=int(MAX_NS_ARCHIVE_SIZE*0.68), k_neighbors=20)
        # self.evaluators["unaligned_novelty"] = NoveltyEvaluatorKD("Unaligned novelty", unaligned_vector, "unaligned_novelty", min_novelty_archive_size=pop_size, max_novelty_archive_size=1000, k_neighbors=20, novelty_threshold=25.)
        self.evaluators.update({"population_saver" : populationSaver,
            "physics" : self.evaluators["physics"],
            "aligned_novelty" : NoveltyEvaluatorKD("Aligned Novelty", backup_path, "aligned_novelty", aligned_vector_gpu, 
                                                    min_novelty_archive_size=pop_size//2, max_novelty_archive_size=500, 
                                                    k_neighbors=20, novelty_threshold=.5), 
            "unaligned_novelty" : NoveltyEvaluatorKD("Unaligned novelty", backup_path, "unaligned_novelty", unaligned_vector,
                                                     min_novelty_archive_size=pop_size, max_novelty_archive_size=1000, 
                                                     k_neighbors=20, novelty_threshold=25.),
            "genotype_diversity_evaluator" : genotypeDiversityEvaluator,
            "genotype_diversity_extractor" : GenotypeDiversityExtractor(genotypeDiversityEvaluator)})


    def _extractObjectives(self, x: SoftBot) -> List[float]:
        return [-x.fitness]

    # def _extractConstraints(self, x: SoftBot) -> List[float]:
    #     return [-x.num_voxels + 0.1*x.genotype.ds_size, x.num_voxels - 0.9*x.genotype.ds_size, -x.active + 0.1*x.num_voxels, x.active - 0.8*x.num_voxels]


class QualityNoveltySoftbotProblem(BaseSoftbotProblem):

    def __init__(self, physics_evaluator : BaseSoftBotPhysicsEvaluator, pop_size : int, backup_path : str):
        super().__init__(physics_evaluator, n_var=1, n_obj=2)
        genotypeDiversityEvaluator = GenotypeDiversityEvaluator()
        populationSaver = PopulationSaver(backup_path)
        # self.evaluators["unaligned_novelty"] = NoveltyEvaluator(unaligned_distance_metric, "unaligned_novelty", is_valid_func=is_valid_func, min_novelty_archive_size=pop_size, max_novelty_archive_size=int(MAX_NS_ARCHIVE_SIZE*0.68), k_neighbors=20)
        # self.evaluators["unaligned_novelty"] = NoveltyEvaluatorKD("Unaligned novelty", unaligned_vector, "unaligned_novelty", min_novelty_archive_size=pop_size, max_novelty_archive_size=1000, k_neighbors=20, novelty_threshold=25.)
        self.evaluators.update({"population_saver" : populationSaver,
            "physics" : self.evaluators["physics"],
            "aligned_novelty" : NoveltyEvaluatorKD("Aligned Novelty", backup_path, "aligned_novelty", aligned_vector_gpu, min_novelty_archive_size=pop_size//2, max_novelty_archive_size=500, k_neighbors=20, novelty_threshold=.5), 
            "unaligned_novelty" : NoveltyEvaluatorKD("Unaligned novelty", backup_path, "unaligned_novelty", unaligned_vector, min_novelty_archive_size=pop_size, max_novelty_archive_size=1000, k_neighbors=20, novelty_threshold=25.),
            "genotype_diversity_evaluator" : genotypeDiversityEvaluator,
            "genotype_diversity_extractor" : GenotypeDiversityExtractor(genotypeDiversityEvaluator)})


    def _extractObjectives(self, x: SoftBot) -> List[float]:
        return [-x.fitness, -x.unaligned_novelty]
    
    # def _extractConstraints(self, x: SoftBot) -> List[float]:
    #     return [-x.num_voxels + 0.1*x.genotype.ds_size, x.num_voxels - 0.9*x.genotype.ds_size, -x.active + 0.1*x.num_voxels, x.active - 0.8*x.num_voxels]

class NSLCSoftbotProblem(BaseSoftbotProblem):

    def __init__(self, physics_evaluator : BaseSoftBotPhysicsEvaluator, pop_size : int, backup_path : str):
        super().__init__(physics_evaluator, n_var=1, n_obj=2)
        genotypeDiversityEvaluator = GenotypeDiversityEvaluator()
        populationSaver = PopulationSaver(backup_path)
        # self.evaluators.update({"unaligned_novelty" : NoveltyEvaluator(unaligned_distance_metric, "unaligned_novelty", nslc_neighbors_name="unaligned_neighbors", is_valid_func=is_valid_func, min_novelty_archive_size=pop_size, max_novelty_archive_size=int(MAX_NS_ARCHIVE_SIZE*0.68), k_neighbors=20),
        #                     "nslc_quality" : NSLCQuality()})
        # self.evaluators.update({"unaligned_nslc" : NSLCEvaluator("Unaligned NSLC", unaligned_vector, "unaligned_novelty", "nslc_quality", "fitness", min_novelty_archive_size=pop_size//2, max_novelty_archive_size=1000, k_neighbors=20, novelty_threshold=25.)})
        self.evaluators.update({"population_saver" : populationSaver,
            "physics" : self.evaluators["physics"],
            "aligned_novelty" : NoveltyEvaluatorKD("Aligned Novelty", backup_path, "aligned_novelty", aligned_vector_gpu, 
                                                    min_novelty_archive_size=pop_size//2, max_novelty_archive_size=500, 
                                                    k_neighbors=20, novelty_threshold=.5), 
            "unaligned_nslc" : NSLCEvaluator("Unaligned NSLC", backup_path, "unaligned_novelty", unaligned_vector, 
                                            "nslc_quality", "fitness", min_novelty_archive_size=pop_size//2, max_novelty_archive_size=1000, 
                                            k_neighbors=20, novelty_threshold=25.),
            "genotype_diversity_evaluator" : genotypeDiversityEvaluator,
            "genotype_diversity_extractor" : GenotypeDiversityExtractor(genotypeDiversityEvaluator)})

    def _extractObjectives(self, x: SoftBot) -> List[float]:
        return [-x.nslc_quality, -x.unaligned_novelty]

    # def _extractConstraints(self, x: SoftBot) -> List[float]:
    #     return [-x.num_voxels + 0.1*x.genotype.ds_size, x.num_voxels - 0.9*x.genotype.ds_size, -x.active + 0.1*x.num_voxels, x.active - 0.8*x.num_voxels]

class MNSLCSoftbotProblem(BaseSoftbotProblem):

    def __init__(self, physics_evaluator : BaseSoftBotPhysicsEvaluator, pop_size : int, backup_path : str):
        super().__init__(physics_evaluator, n_var=1, n_obj=3)
        genotypeDiversityEvaluator = GenotypeDiversityEvaluator()
        populationSaver = PopulationSaver(backup_path)
        # self.evaluators.update({"aligned_novelty" : NoveltyEvaluator(aligned_distance_metric, "aligned_novelty", is_valid_func=is_valid_func, min_novelty_archive_size=pop_size, max_novelty_archive_size=int(MAX_NS_ARCHIVE_SIZE*0.68), k_neighbors=20), 
        #     "unaligned_novelty" : NoveltyEvaluator(unaligned_distance_metric, "unaligned_novelty", nslc_neighbors_name="unaligned_neighbors", is_valid_func=is_valid_func, min_novelty_archive_size=pop_size, max_novelty_archive_size=int(MAX_NS_ARCHIVE_SIZE*0.68), k_neighbors=20),
        #     "nslc_quality" : NSLCQuality()})
        # self.evaluators.update({"aligned_novelty" : NoveltyEvaluatorKD("Aligned Novelty", aligned_vector, "aligned_novelty", min_novelty_archive_size=pop_size, max_novelty_archive_size=500, k_neighbors=20), 
        #             "unaligned_nslc" : NSLCEvaluator("Unaligned NSLC", unaligned_vector, "unaligned_novelty", "nslc_quality", "fitness", min_novelty_archive_size=pop_size//2, max_novelty_archive_size=1000, k_neighbors=20, novelty_threshold=25.)})
        self.evaluators.update({"population_saver" : populationSaver,
            "physics" : self.evaluators["physics"],
            "aligned_novelty" : NoveltyEvaluatorKD("Aligned Novelty", backup_path, "aligned_novelty", aligned_vector, 
                                                    min_novelty_archive_size=pop_size//2, max_novelty_archive_size=500, 
                                                    k_neighbors=20, novelty_threshold=.5), 
            "unaligned_nslc" : NSLCEvaluator("Unaligned NSLC", backup_path, "unaligned_novelty", unaligned_vector, 
                                            "nslc_quality", "fitness", min_novelty_archive_size=pop_size//2, 
                                            max_novelty_archive_size=1000, k_neighbors=20, novelty_threshold=25.),
            "genotype_diversity_evaluator" : genotypeDiversityEvaluator,
            "genotype_diversity_extractor" : GenotypeDiversityExtractor(genotypeDiversityEvaluator)})

    def _extractObjectives(self, x: SoftBot) -> List[float]:
        return [-x.nslc_quality, -x.aligned_novelty, -x.unaligned_novelty]

    # def _extractConstraints(self, x: SoftBot) -> List[float]:
    #     return [-x.num_voxels + 0.1*x.genotype.ds_size, x.num_voxels - 0.9*x.genotype.ds_size, -x.active + 0.1*x.num_voxels, x.active - 0.8*x.num_voxels]

class MNSLCSoftbotProblemGPU(BaseSoftbotProblem):

    def __init__(self, physics_evaluator : BaseSoftBotPhysicsEvaluator, pop_size : int, backup_path : str):
        super().__init__(physics_evaluator, n_var=1, n_obj=3)
        # self.evaluators.update({"aligned_novelty" : NoveltyEvaluator(aligned_distance_metric_gpu, "aligned_novelty", is_valid_func=is_valid_func, min_novelty_archive_size=pop_size, max_novelty_archive_size=int(MAX_NS_ARCHIVE_SIZE*0.68), k_neighbors=20), 
        #                     "unaligned_novelty" : NoveltyEvaluator(unaligned_distance_metric, "unaligned_novelty", nslc_neighbors_name="unaligned_neighbors", is_valid_func=is_valid_func, min_novelty_archive_size=pop_size, max_novelty_archive_size=int(MAX_NS_ARCHIVE_SIZE*0.68), k_neighbors=20),
        #                     "nslc_quality" : NSLCQuality()})
        genotypeDiversityEvaluator = GenotypeDiversityEvaluator()
        populationSaver = PopulationSaver(backup_path)
        self.evaluators.update({"population_saver" : populationSaver,
            "physics" : self.evaluators["physics"],
            "aligned_novelty" : NoveltyEvaluatorKD("Aligned Novelty", backup_path, "aligned_novelty", aligned_vector_gpu, 
                                                    min_novelty_archive_size=pop_size//2, max_novelty_archive_size=500, 
                                                    k_neighbors=20, novelty_threshold=.5), 
            "unaligned_nslc" : NSLCEvaluator("Unaligned NSLC", backup_path, "unaligned_novelty", unaligned_vector, "nslc_quality", "fitness", min_novelty_archive_size=pop_size//2, max_novelty_archive_size=1000, k_neighbors=20, novelty_threshold=25.),
            "genotype_diversity_evaluator" : genotypeDiversityEvaluator,
            "genotype_diversity_extractor" : GenotypeDiversityExtractor(genotypeDiversityEvaluator)})
        

    def _extractObjectives(self, x: SoftBot) -> List[float]:
        return [-x.nslc_quality, -x.aligned_novelty, -x.unaligned_novelty]

    # def _extractConstraints(self, x: SoftBot) -> List[float]:
    #     return [-x.num_voxels + 0.1*x.genotype.ds_size, x.num_voxels - 0.9*x.genotype.ds_size, -x.active + 0.1*x.num_voxels, x.active - 0.9*x.num_voxels]