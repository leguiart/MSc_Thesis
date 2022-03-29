import copy
import random
import os
import sys
import inspect
import numpy as np

from pymoo.core.crossover import ElementwiseCrossover, Crossover


class DummySoftbotCrossover(Crossover):
    def __init__(self):
        # define the crossover: number of parents and number of offsprings
        super().__init__(2,2,0.0)
    
    def do(self, problem, pop, parents, **kwargs):
        return pop