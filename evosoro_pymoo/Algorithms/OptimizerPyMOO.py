
import numpy as np
import subprocess as sub
import os
import sys
import logging

from pymoo.core.callback import Callback
from pymoo.core.evaluator import set_cv
from pymoo.core.algorithm import Algorithm
from common.IAnalytics import IAnalytics


# sys.path.append(os.getcwd() + "/..")
from evosoro.tools.logging import make_gen_directories, initialize_folders, write_gen_stats
from evosoro.tools.algorithms import Optimizer
from evosoro_pymoo.Problems.SoftbotProblem import BaseSoftbotProblem
from common.Utils import timeit
from evosoro_pymoo.common.IStart import IStarter

logger = logging.getLogger(f"__main__.{__name__}")

class PopulationBasedOptimizerPyMOO(Optimizer, IStarter):
    def __init__(self, sim, env, algorithm : Algorithm, problem : BaseSoftbotProblem, analytics : IAnalytics = None):
        Optimizer.__init__(self, sim, env)

        self.algorithm = algorithm
        self.problem = problem
        self.analytics = analytics

    def start(self):
        self.problem.start()
        self.analytics.start() 

    @timeit
    def run(self, evosoro_pop, save_pareto=False,
            save_nets=False, continued_from_checkpoint=False):

        # while the algorithm has not terminated
        while self.algorithm.has_next():

            self.ask_tell(evosoro_pop)
        
        # obtain the result objective from the algorithm
        res = self.algorithm.result()

        return res

    @timeit
    def ask_tell(self, evosoro_pop):

        
        logger.info("Now creating new population")
        # ask the algorithm for the next solution to be evaluated
        # basically, pymoo generates a new population by applying the variation operators (mutation and crossover) from selected parent population
        pop = self.algorithm.ask()


        for i in range(len(evosoro_pop)):
            evosoro_pop.individuals[i] = pop[i].X


        # evaluate the individuals using the algorithm's evaluator (necessary to count evaluations for termination)
        logger.info("Starting individuals evaluation")
        mat_pop = []
        for ind in pop:
            mat_pop += [np.array([ind])]
        mat_pop = np.array(mat_pop)
        F, G, _ = self.algorithm.evaluator.eval(self.problem, mat_pop)
        pop.set("F", F)
        pop.set("G", G)
        #pop.set("CV", CV)
        set_cv(pop)

        # Extract analytics
        if self.analytics is not None:
            logger.debug("Collecting analytics data")
            self.analytics.notify(pop, self.problem)
            logger.debug("Finished collecting analytics data")

        logger.info("Individuals evaluation finished")  # record total eval time in log

        # returned the evaluated individuals which have been evaluated or even modified
        self.algorithm.tell(infills=pop)

