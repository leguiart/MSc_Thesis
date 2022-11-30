
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
    def evaluate(self, X : list, *args, **kwargs):
        self.genotypeDistanceEvaluator.evaluate(X)

        for i in range(len(X)):
            gene_diversity = []
            morpho_diversity = []
            control_diversity = []
            for j in range(len(X)):
                if i != j:
                    ind1_id = X[i].id
                    ind2_id = X[j].id

                    gene_diversity += [self.genotypeDistanceEvaluator[ind1_id, ind2_id][2]]
                    morpho_diversity += [self.genotypeDistanceEvaluator[ind1_id, ind2_id][1]]
                    control_diversity += [self.genotypeDistanceEvaluator[ind1_id, ind2_id][0]]
                    
            gene_diversity = np.array(gene_diversity)
            morpho_diversity = np.array(morpho_diversity)
            control_diversity = np.array(control_diversity)
            X[i].gene_diversity = gene_diversity.mean()
            X[i].control_gene_div = control_diversity.mean()
            X[i].morpho_gene_div = morpho_diversity.mean()


    def clean(self, *args, **kwargs):
        self.genotypeDistanceEvaluator.clean(args[0], kwargs['pop_size'])

            
 