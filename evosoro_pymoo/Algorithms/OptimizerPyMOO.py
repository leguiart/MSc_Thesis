
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
        # self.num_env_cycles = 0
        # self.autosuspended = False
        # self.max_gens = None
        # self.directory = None
        # self.name = None
        # self.num_random_inds = 0
        self.problem = problem
        self.analytics = analytics

    def start(self):
        self.problem.start()
        self.analytics.start() 

    @timeit
    def run(self, evosoro_pop, max_hours_runtime=29, max_gens=3000, num_random_individuals=1, num_env_cycles=0,
            checkpoint_every=100, save_pareto=False,
            save_nets=False, continued_from_checkpoint=False, new_run = True):

        # if self.autosuspended:
        #     sub.call("rm %s/AUTOSUSPENDED" % self.problem.evaluators["physics"].run_directory, shell=True)

        # self.autosuspended = False
        # self.max_gens = max_gens  # can add additional gens through checkpointing

        #if not continued_from_checkpoint:  # generation zero
        # if new_run:
        #     self.directory = self.problem.evaluators["physics"].run_directory
        #     self.name = self.problem.evaluators["physics"].run_name
        #     self.num_random_inds = num_random_individuals
        #     self.num_env_cycles = num_env_cycles
        #     initialize_folders(evosoro_pop, self.problem.evaluators["physics"].run_directory, self.problem.evaluators["physics"].run_name, save_nets, save_lineages=self.problem.evaluators["physics"].save_lineages)
        #     make_gen_directories(evosoro_pop, self.problem.evaluators["physics"].run_directory, self.problem.evaluators["physics"].save_vxa_every, save_nets)
        #     sub.call("touch {}/RUNNING".format(self.directory), shell=True)
        # self.evaluate(self.sim, self.env[self.curr_env_idx], self.pop, self.problem.print_log, save_vxa_every, self.directory,
        #               self.name, max_eval_time, time_to_try_again, save_lineages)
        # self.select(self.pop)  # only produces dominated_by stats, no selection happening (population not replaced)
        # write_gen_stats(self.pop, self.directory, self.name, save_vxa_every, save_pareto, save_nets,
        #                 save_lineages=save_lineages)
        

        # while the algorithm has not terminated
        while self.algorithm.has_next():

            self.ask_tell(evosoro_pop)
        
        # obtain the result objective from the algorithm
        res = self.algorithm.result()

        # if not self.autosuspended:  # print end of run stats
        #     logger.info("Finished {0} generations".format(self.algorithm.n_gen + 1))
        #     logger.info("DONE!")
        #     sub.call("touch {0}/RUN_FINISHED && rm {0}/RUNNING".format(self.directory), shell=True)
        
        return res

    @timeit
    def ask_tell(self, evosoro_pop):

        # if self.algorithm.n_gen % checkpoint_every == 0:
        #     self.problem.print_log.message("Saving checkpoint at generation {0}".format(self.algorithm.n_gen+1), timer_name="start")
        #     self.save_checkpoint(self.directory, self.algorithm.n_gen)

        # if self.elapsed_time(units="h") > max_hours_runtime:
        #     self.autosuspended = True
        #     self.problem.print_log.message("Autosuspending at generation {0}".format(self.algorithm.n_gen+1), timer_name="start")
        #     self.save_checkpoint(self.directory, self.algorithm.n_gen)
        #     sub.call("touch {0}/AUTOSUSPENDED && rm {0}/RUNNING".format(self.directory), shell=True)
        #     break
        
        # self.problem.evaluators["physics"].n_gen = self.algorithm.n_gen if self.algorithm.n_gen != None else 1
        logger.info("Now creating new population")
        # self.problem.evaluators["physics"].update_env()
        
        # ask the algorithm for the next solution to be evaluated
        # basically, pymoo generates a new population by applying the variation operators (mutation and crossover)
        pop = self.algorithm.ask()

        # logger.info("Creating folders structure for this generation")
        # evosoro_pop.gen = self.problem.evaluators["physics"].n_gen
        for i in range(len(evosoro_pop)):
            evosoro_pop.individuals[i] = pop[i].X
        # if new_run:
        #     make_gen_directories(evosoro_pop, self.directory, self.problem.evaluators["physics"].save_vxa_every, save_nets)

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
            logger.debug("Finishing collecting analytics data")

        logger.info("Individuals evaluation finished")  # record total eval time in log

        # returned the evaluated individuals which have been evaluated or even modified
        self.algorithm.tell(infills=pop)

        # print population to stdout and save all individual data
        #self.problem.print_log.message("Saving statistics")
        #write_gen_stats(pop, self.directory, self.name, self.problem.save_vxa_every, save_pareto, save_nets,
        #save_lineages=self.problem.save_lineages)