
import copy
import numpy as np
import pickle
import logging

from pymoo.core.evaluator import set_cv
from pymoo.core.algorithm import Algorithm
from pymoo.core.population import Population


from evosoro.base import Env, Sim
from evosoro.tools.algorithms import Optimizer
from evosoro_pymoo.Problems.SoftbotProblem import BaseSoftbotProblem
from evosoro_pymoo.common.ICheckpoint import ICheckpoint
from common.Utils import readFromDill, saveToDill, saveToPickle, timeit
from common.IAnalytics import IAnalytics
from evosoro_pymoo.common.IStart import IStarter

logger = logging.getLogger(f"__main__.{__name__}")

class PopulationBasedOptimizerPyMOO(Optimizer, ICheckpoint, IStarter):
    def __init__(self, sim : Sim, env : Env, 
                algorithm : Algorithm, 
                problem : BaseSoftbotProblem, 
                analytics : IAnalytics = None, 
                save_checkpoint = False,
                checkpoint_path = '.'):

        Optimizer.__init__(self, sim, env)
        self.algorithm = algorithm
        self.problem = problem
        self.analytics = analytics
        self.save_to_checkpoint = save_checkpoint
        self.checkpoint_path = checkpoint_path

    def start(self, **kwargs):
        resuming_run = kwargs["resuming_run"]
        if resuming_run:
            self.algorithm = readFromDill(f"{self.checkpoint_path}/algorithm_checkpoint.pickle")
            self.analytics  = self.analytics.file_recovery()

        self.problem.start(resuming_run = resuming_run)
        self.analytics.start(resuming_run = resuming_run) 

    def backup(self):
        saveToPickle(f"{self.checkpoint_path}/algorithm_checkpoint.pickle", self.algorithm)
        self.problem.backup()
        self.analytics.backup()

    @timeit
    def run(self):

        # while the algorithm has not terminated
        while self.algorithm.has_next():
            self.next()
    
        # obtain the result objective from the algorithm
        res = self.algorithm.result()

        return res

    @timeit
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
        mat_pop = []
        for ind in pop:
            mat_pop += [np.array([ind])]
        mat_pop = np.array(mat_pop)

        F, G, _ = self.algorithm.evaluator.eval(self.problem, mat_pop, pop_size = len(children_pop))
        pop.set("F", F)
        pop.set("G", G)
        set_cv(pop)

        # Extract analytics
        if self.analytics is not None:
            logger.debug("Collecting analytics data")
            self.analytics.notify(children_pop, self.problem)
            logger.debug("Finished collecting analytics data")

        logger.info("Individuals evaluation finished")  # record total eval time in log

        # returned the evaluated individuals which have been evaluated or even modified
        # here we produce the next generation parent population by applying a survival technique
        # (elitism, ranking, ordering, replacement, etc.)
        self.algorithm.tell(infills=children_pop)

        if self.save_to_checkpoint:
            self.backup()

