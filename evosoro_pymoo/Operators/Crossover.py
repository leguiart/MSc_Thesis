
import numpy as np

from pymoo.core.crossover import Crossover


class DummySoftbotCrossover(Crossover):
    def __init__(self):
        # define the crossover: number of parents and number of offsprings
        super().__init__(2,2,0.0)
    
    def do(self, problem, pop, parents, **kwargs):
        flattened_parents_indxs = parents.reshape(len(pop))

        return pop[flattened_parents_indxs]