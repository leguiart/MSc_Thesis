
import copy
import numpy as np
import pickle
import logging

from pymoo.core.evaluator import set_cv
from pymoo.core.algorithm import Algorithm
from pymoo.core.population import Population
from pymoo.core.problem import Problem


logger = logging.getLogger(f"__main__.{__name__}")

class PopulationBasedOptimizer:
    def __init__(self,
                algorithm : Algorithm, 
                problem : Problem):

        self.algorithm = algorithm
        self.problem = problem

    def run(self):

        # while the algorithm has not terminated
        while self.algorithm.has_next():
            self.next()
    
        # obtain the results from the algorithm
        return self.algorithm.result()

    def next(self):

        
        logger.info("Now creating new population")
        # ask the algorithm for the next solution to be evaluated
        # basically, pymoo generates a new population of children by applying 
        # the variation operators (mutation and crossover) 
        # to the set of parents which are selected (binary tournament, proportional, etc.)
        # from the parent population
        children_pop = self.algorithm.ask()
        if self.algorithm.is_initialized:
            parent_pop = self.algorithm.pop 
            pop = Population.merge(parent_pop, children_pop)
        else:
            pop = children_pop
            
        # evaluate the individuals using the algorithm's evaluator (necessary to count evaluations for termination)
        logger.info("Starting individuals evaluation")
        self.algorithm.evaluator.eval(self.problem, pop, pop_size = len(children_pop), n_gen = self.algorithm.n_gen, skip_already_evaluated=False)
        logger.info("Individuals evaluation finished")

        # returned the evaluated individuals which have been evaluated or even modified
        # here we produce the next generation parent population by applying a survival technique
        # (elitism, ranking, ordering, replacement, etc.)
        self.algorithm.tell(infills=children_pop)