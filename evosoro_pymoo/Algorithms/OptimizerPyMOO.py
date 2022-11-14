
import copy
import numpy as np
import pickle
import logging

# from pymoo.core.evaluator import set_cv
from pymoo.core.algorithm import Algorithm
from pymoo.core.population import Population


from evosoro.base import Env, Sim
from evosoro.tools.algorithms import Optimizer
from evosoro_pymoo.Problems.SoftbotProblem import BaseSoftbotProblem
from evosoro_pymoo.common.ICheckpoint import ICheckpoint
from common.Utils import readFromDill, saveToDill, saveToPickle, timeit
from evosoro_pymoo.common.IAnalytics import IAnalytics
from evosoro_pymoo.common.IStart import IStarter
from evosoro_pymoo.common.IStateCleaner import IStateCleaning

logger = logging.getLogger(f"__main__.{__name__}")

class PopulationBasedOptimizerPyMOO(Optimizer, ICheckpoint, IStarter):
    def __init__(self, sim : Sim, env : Env, 
                algorithm : Algorithm, 
                problem : BaseSoftbotProblem, 
                analytics : IAnalytics = None, 
                save_checkpoint : bool = False, 
                save_every : int = 1,
                checkpoint_path : str = '.',
                save_networks : bool = False):

        Optimizer.__init__(self, sim, env)
        self.algorithm = algorithm
        self.problem = problem
        self.analytics = analytics
        self.save_to_checkpoint = save_checkpoint
        self.save_checkpoint_every = save_every
        self.checkpoint_path = checkpoint_path
        self.save_networks = save_networks

    def start(self, **kwargs):
        resuming_run = kwargs["resuming_run"]
        if resuming_run:
            self.algorithm = readFromDill(f"{self.checkpoint_path}/algorithm_checkpoint.pickle")
            self.analytics  = self.analytics.file_recovery()
            # gen = min(self.analytics.actual_generation, self.algorithm.n_gen)
            # self.algorithm.n_gen = gen
            # self.analytics.actual_generation = gen

        self.problem.start(**kwargs)
        self.analytics.start(**kwargs) 

    def backup(self, *args, **kwargs):
        saveToPickle(f"{self.checkpoint_path}/algorithm_checkpoint.pickle", self.algorithm)
        self.problem.backup()
        self.analytics.backup()


    @timeit
    def run(self):

        # while the algorithm has not terminated
        while self.algorithm.has_next():
            self.next()
    
        # obtain the result objective from the algorithm
        result_set = {}
        result_set['res'] = self.algorithm.result()
        result_set.update(self.problem.results())

        return result_set

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
            logger.debug(f"Parent population size: {len(parent_pop)}. Child population size: {len(children_pop)}")
        else:
            pop = children_pop
            
        # evaluate the individuals using the algorithm's evaluator (necessary to count evaluations for termination)
        logger.info("Starting individuals evaluation")
        mat_pop = []
        lst_pop = []
        for ind in pop:
            lst_pop += [ind.X for ind in pop]
            mat_pop += [np.array([ind])]
        mat_pop = np.array(mat_pop)

        if issubclass(type(self.problem), IStateCleaning):
            if self.algorithm.is_initialized:
                self.problem.clean(lst_pop, pop_size = len(parent_pop))


        F, G, _ = self.algorithm.evaluator.eval(self.problem, mat_pop, pop_size = len(children_pop), n_gen = self.algorithm.n_gen)
        pop.set("F", F)
        pop.set("G", G)
        # set_cv(pop)
        
        # Save networks
        if self.save_networks:
            if self.algorithm.is_initialized:
                pop_networks = [ind.X for ind in pop]
                saveToPickle(f"{self.checkpoint_path}/Gen_{self.algorithm.n_gen:04d}/Gen_{self.algorithm.n_gen:04d}_population.pickle", pop_networks)

        # Extract analytics
        if self.analytics is not None:
            logger.debug("Collecting analytics data")
            if self.algorithm.is_initialized:
                self.analytics.notify(self.algorithm, pop = parent_pop, child_pop = children_pop)
            else:
                self.analytics.notify(self.algorithm, pop = pop, child_pop = pop)
            logger.debug("Finished collecting analytics data")

        logger.info("Individuals evaluation finished")  # record total eval time in log

        # returned the evaluated individuals which have been evaluated or even modified
        # here we produce the next generation parent population by applying a survival technique
        # (elitism, ranking, ordering, replacement, etc.)
        self.algorithm.tell(infills=children_pop)

        if self.save_to_checkpoint and self.algorithm.n_gen % self.save_checkpoint_every == 0:
            self.backup()

