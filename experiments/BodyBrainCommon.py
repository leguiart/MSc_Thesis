#!/usr/bin/python

import numpy as np
import uuid
import subprocess as sub
import os
import sys
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.population import Population

# Appending repo's root dir in the python path to enable subsequent imports
from evosoro_pymoo.Algorithms.OptimizerPyMOO import PopulationBasedOptimizerPyMOO
from Constants import *
from evosoro_pymoo.Operators.Crossover import DummySoftbotCrossover 
from evosoro_pymoo.Operators.Mutation import SoftbotMutation
from Analytics.Utils import setRandomSeed, readFromJson, writeToJson, QD_Analytics
sys.path.append(os.getcwd() + "/..")
from evosoro.base import Sim, Env, ObjectiveDict
from evosoro.softbot import Population as SoftbotPopulation, Genotype, Phenotype
from evosoro_pymoo.Problems.SoftbotProblem import BaseSoftbotProblem

sub.call("rm ./voxelyze", shell=True)
sub.call("cp ../" + VOXELYZE_VERSION + "/voxelyzeMain/voxelyze .", shell=True)  # Making sure to have the most up-to-date version of the Voxelyze physics engine


def runBodyBrain(runs : int, pop_size : int, max_gens : int, seeds_json : str, 
                analytics_json : str, objective_dict : ObjectiveDict, 
                softbot_problem_cls : BaseSoftbotProblem, 
                genotype_cls : Genotype, phenotype_cls : Phenotype):
    runToSeedMapping = readFromJson(seeds_json)
    runToAnalyticsMapping = readFromJson(analytics_json)


    for run in range(runs):
        # Setting random seed
        if run + 1 not in runToSeedMapping:
            runToSeedMapping[run + 1] = uuid.uuid4().int & (1<<32)-1
            writeToJson(seeds_json, runToSeedMapping)

        setRandomSeed(runToSeedMapping[run + 1])
        
        # Setting up the simulation object
        sim = Sim(dt_frac=DT_FRAC, simulation_time=SIM_TIME, fitness_eval_init_time=INIT_TIME)

        # Setting up the environment object
        env = Env(sticky_floor=0, time_between_traces=0)


        # Initializing a population of SoftBots
        my_pop = SoftbotPopulation(objective_dict, genotype_cls, phenotype_cls, pop_size=pop_size)
        pop = Population.new("X", my_pop)

        #Setting up Softbot optimization problem
        softbot_problem = softbot_problem_cls(sim, env, SAVE_POPULATION_EVERY, RUN_DIR_SO, RUN_NAME_SO + str(run + 1), MAX_EVAL_TIME, TIME_TO_TRY_AGAIN, SAVE_LINEAGES, objective_dict)

        # Setting up our optimization
        algorithm = NSGA2(pop_size=pop_size, sampling=np.array(my_pop.individuals), mutation=SoftbotMutation(), crossover=DummySoftbotCrossover(), eliminate_duplicates=False)
        algorithm.setup(softbot_problem, termination=('n_gen', max_gens), seed = runToSeedMapping[run + 1])
        analytics = QD_Analytics()
        my_optimization = PopulationBasedOptimizerPyMOO(sim, env, algorithm, softbot_problem, analytics)

        if run + 1 not in runToAnalyticsMapping:
            # start optimization
            my_optimization.run(my_pop, max_hours_runtime=MAX_TIME, max_gens=max_gens, num_random_individuals=NUM_RANDOM_INDS, checkpoint_every=CHECKPOINT_EVERY, new_run = run == 0)
            runToAnalyticsMapping[run + 1] = analytics.qd_history
            writeToJson(analytics_json, runToAnalyticsMapping)

# if __name__ == "__main__":
#     main(sys.argv[1:])
