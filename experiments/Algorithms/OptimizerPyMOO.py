import numpy as np
import subprocess as sub
import os
import sys
from pymoo.core.evaluator import set_cv
from pymoo.core.algorithm import Algorithm



sys.path.append(os.getcwd() + "/..")
from evosoro.tools.logging import make_gen_directories, initialize_folders, write_gen_stats
from evosoro.tools.algorithms import Optimizer

class PopulationBasedOptimizerPyMOO(Optimizer):
    def __init__(self, sim, env, algorithm : Algorithm, problem, analytics = None):
        Optimizer.__init__(self, sim, env)
        self.algorithm = algorithm
        self.num_env_cycles = 0
        self.autosuspended = False
        self.max_gens = None
        self.directory = None
        self.name = None
        self.num_random_inds = 0
        self.problem = problem
        self.analytics = analytics

    # def update_env(self):
    #     if self.num_env_cycles > 0:
    #         switch_every = self.max_gens / float(self.num_env_cycles)
    #         self.curr_env_idx = int(self.algorithm.n_gen / switch_every % len(self.problem.env))
    #         print (" Using environment {0} of {1}".format(self.curr_env_idx+1, len(self.problem.env)))

    def run(self, evosoro_pop, max_hours_runtime=29, max_gens=3000, num_random_individuals=1, num_env_cycles=0,
            checkpoint_every=100, save_pareto=False,
            save_nets=False, continued_from_checkpoint=False, new_run = True):

        if self.autosuspended:
            sub.call("rm %s/AUTOSUSPENDED" % self.problem.directory, shell=True)

        self.autosuspended = False
        self.max_gens = max_gens  # can add additional gens through checkpointing

        self.problem.print_log.add_timer("evaluation")
        self.start_time = self.problem.print_log.timers["start"]  # sync start time with logging

        # sub.call("clear", shell=True)

        #if not continued_from_checkpoint:  # generation zero
        if new_run:
            self.directory = self.problem.directory
            self.name = self.problem.name
            self.num_random_inds = num_random_individuals
            self.num_env_cycles = num_env_cycles
            initialize_folders(evosoro_pop, self.problem.directory, self.problem.name, save_nets, save_lineages=self.problem.save_lineages)
            make_gen_directories(evosoro_pop, self.problem.directory, self.problem.save_vxa_every, save_nets)
            sub.call("touch {}/RUNNING".format(self.directory), shell=True)
        # self.evaluate(self.sim, self.env[self.curr_env_idx], self.pop, self.problem.print_log, save_vxa_every, self.directory,
        #               self.name, max_eval_time, time_to_try_again, save_lineages)
        # self.select(self.pop)  # only produces dominated_by stats, no selection happening (population not replaced)
        # write_gen_stats(self.pop, self.directory, self.name, save_vxa_every, save_pareto, save_nets,
        #                 save_lineages=save_lineages)
        

        # until the algorithm has not terminated
        while self.algorithm.has_next():

            # if self.algorithm.n_gen % checkpoint_every == 0:
            #     self.problem.print_log.message("Saving checkpoint at generation {0}".format(self.algorithm.n_gen+1), timer_name="start")
            #     self.save_checkpoint(self.directory, self.algorithm.n_gen)

            # if self.elapsed_time(units="h") > max_hours_runtime:
            #     self.autosuspended = True
            #     self.problem.print_log.message("Autosuspending at generation {0}".format(self.algorithm.n_gen+1), timer_name="start")
            #     self.save_checkpoint(self.directory, self.algorithm.n_gen)
            #     sub.call("touch {0}/AUTOSUSPENDED && rm {0}/RUNNING".format(self.directory), shell=True)
            #     break
           
            # ask the algorithm for the next solution to be evaluated
            self.problem.n_gen = self.algorithm.n_gen if self.algorithm.n_gen != None else 0
            self.problem.print_log.message("Now creating new population")
            self.problem.update_env()
            pop = self.algorithm.ask()

            self.problem.print_log.message("Creating folders structure for this generation")
            evosoro_pop.gen = self.problem.n_gen
            for i in range(len(evosoro_pop)):
                evosoro_pop.individuals[i] = pop[i].X
            if new_run:
                make_gen_directories(evosoro_pop, self.directory, self.problem.save_vxa_every, save_nets)

            # evaluate fitness
            # evaluate the individuals using the algorithm's evaluator (necessary to count evaluations for termination)
            self.problem.print_log.message("Starting fitness evaluation", timer_name="start")
            self.problem.print_log.reset_timer("evaluation")
            # pop = np.reshape(pop, (len(pop), 1))
            mat_pop = []
            for ind in pop:
                mat_pop += [np.array([ind])]
            mat_pop = np.array(mat_pop)
            F, G, _ = self.algorithm.evaluator.eval(self.problem, mat_pop)
            pop.set("F", F)
            pop.set("G", G)
            #pop.set("CV", CV)
            set_cv(pop)
            if self.analytics is not None:
                self.analytics.notify(pop)
            self.problem.print_log.message("Fitness evaluation finished", timer_name="evaluation")  # record total eval time in log
            # returned the evaluated individuals which have been evaluated or even modified
            self.algorithm.tell(infills=pop)
            # print population to stdout and save all individual data
            #self.problem.print_log.message("Saving statistics")
            #write_gen_stats(pop, self.directory, self.name, self.problem.save_vxa_every, save_pareto, save_nets,
                            #save_lineages=self.problem.save_lineages)
            
        
        # obtain the result objective from the algorithm
        res = self.algorithm.result()

        if not self.autosuspended:  # print end of run stats
            self.problem.print_log.message("Finished {0} generations".format(self.algorithm.n_gen + 1))
            self.problem.print_log.message("DONE!", timer_name="start")
            sub.call("touch {0}/RUN_FINISHED && rm {0}/RUNNING".format(self.directory), shell=True)
