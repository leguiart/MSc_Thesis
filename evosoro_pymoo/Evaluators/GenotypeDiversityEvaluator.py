
import logging
import os
import numpy as np


from evosoro_pymoo.Evaluators.GenotypeDistanceEvaluator import GenotypeDistanceEvaluator
from evosoro_pymoo.Evaluators.IEvaluator import IEvaluator
from common.Utils import readFromJson, readFromPickle, saveToPickle, timeit
from evosoro_pymoo.common.IStateCleaner import IStateCleaning

logger = logging.getLogger(f"__main__.{__name__}")


class GenotypeDiversityEvaluator(IEvaluator, IStateCleaning, object):

    def __init__(self, orig_size_xyz = (6,6,6)) -> None:
        super().__init__()
        self.genotypeDistanceEvaluator = GenotypeDistanceEvaluator(orig_size_xyz)
        self.gene_div_matrix = []

    def __getitem__(self, n):
        return self.gene_div_matrix[n]

    @timeit
    def evaluate(self, X : list, *args, **kwargs) -> list:
        X = self.genotypeDistanceEvaluator.evaluate(X)
        self.gene_div_matrix = []

        for i in range(len(X)):
            gene_diversity = []
            for j in range(len(X)):
                if i != j:
                    ind1_id = X[i].id
                    ind2_id = X[j].id
                    gene_diversity += [self.genotypeDistanceEvaluator[ind1_id, ind2_id]]
                    
            gene_diversity = np.array(gene_diversity)
            gene_diversity = np.mean(gene_diversity, axis=0)
            self.gene_div_matrix += [list(gene_diversity)]

        return X

    def clean(self, *args, **kwargs):
        self.genotypeDistanceEvaluator.clean(args[0], kwargs['pop_size'])

            
 