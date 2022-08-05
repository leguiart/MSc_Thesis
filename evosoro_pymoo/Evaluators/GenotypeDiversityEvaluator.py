
import logging
import os
import numpy as np


from evosoro_pymoo.Evaluators.GenotypeDistanceEvaluator import GenotypeDistanceEvaluator
from evosoro_pymoo.Evaluators.IEvaluator import IEvaluator
from common.Utils import saveToPickle

logger = logging.getLogger(f"__main__.{__name__}")


class GenotypeDiversityEvaluator(IEvaluator, object):

    def __init__(self, orig_size_xyz = (6,6,6)) -> None:
        super().__init__()
        self.genotypeDistanceEvaluator = GenotypeDistanceEvaluator(orig_size_xyz)
        self.gene_div_matrix = []
        # self.save_checkpoint = save_checkpoint
        # self.base_path = base_path
        # self.checkpoint_path = os.path.join(self.base_path, f"genotypeDiversityEvaluatorCheckpoint.pickle") if save_checkpoint  else ""


    def __getitem__(self, n):
        return self.gene_div_matrix[n]


    def evaluate(self, X : list, *args, **kwargs) -> list:
        X = self.genotypeDistanceEvaluator.evaluate(X)
        self.gene_div_matrix = []

        for i in range(len(X)):
            gene_diversity = []
            for j in range(len(X)):
                if i != j:
                    ind1_md5 = X[i].md5
                    ind2_md5 = X[j].md5
                    gene_diversity += [self.genotypeDistanceEvaluator[ind1_md5, ind2_md5]]
                    
            gene_diversity = np.array(gene_diversity)
            gene_diversity = np.mean(gene_diversity, axis=0)
            self.gene_div_matrix += [list(gene_diversity)]
        # if self.save_checkpoint:
        #     saveToPickle(self.checkpoint_path, self)
        return X

            
 