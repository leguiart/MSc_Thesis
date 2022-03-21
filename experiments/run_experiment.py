#!/usr/bin/python
"""

In this example we evolve running soft robots in a terrestrial environment using a standard version of the physics
engine (_voxcad). After running this program for some time, you can start having a look at some of the evolved
morphologies and behaviors by opening up some of the generated .vxa (e.g. those in
evosoro/evosoro/basic_data/bestSoFar/fitOnly) with ./evosoro/evosoro/_voxcad/release/VoxCad
(then selecting the desired .vxa file from "File -> Import -> Simulation")

The phenotype is here based on a discrete, predefined palette of materials, which are visualized with different colors
when robots are simulated in the GUI.

Materials are identified through a material ID:
0: empty voxel, 1: passiveSoft (light blue), 2: passiveHard (blue), 3: active+ (red), 4:active- (green)

Active+ and Active- voxels are in counter-phase.


Additional References
---------------------

This setup is similar to the one described in:

    Cheney, N., MacCurdy, R., Clune, J., & Lipson, H. (2013).
    Unshackling evolution: evolving soft robots with multiple materials and a powerful generative encoding.
    In Proceedings of the 15th annual conference on Genetic and evolutionary computation (pp. 167-174). ACM.

    Related video: https://youtu.be/EXuR_soDnFo

"""
import numpy as np
import subprocess as sub
import os
import sys, getopt
import uuid
from functools import partial
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.population import Population

from Algorithms.OptimizerPyMOO import PopulationBasedOptimizerPyMOO
from General.Constants import *
from Problems.Variation import DummySoftbotCrossover, SoftbotMutation
from General.Utils import readFromJson, save_json, writeToJson, countFileLines, readFirstJson, QD_Analytics
sys.path.append(os.getcwd() + "/..")# Appending repo's root dir in the python path to enable subsequent imports
from evosoro.base import Sim, Env, ObjectiveDict
from evosoro.tools.utils import count_occurrences
from evosoro.softbot import Population as SoftbotPopulation, Genotype, Phenotype
from Problems.SoftbotProblem import QualitySoftbotProblem, QualityNoveltySoftbotProblem, MNSLCSoftbotProblem, BodyBrainGenotypeIndirect2, SimplePhenotypeIndirect
from BodyBrainCommon import runBodyBrain

sub.call("cp ../evosoro/" + VOXELYZE_VERSION + "/voxelyzeMain/voxelyze .", shell=True)  # Making sure to have the most up-to-date version of the Voxelyze physics engine


def main(argv):
    try:
        opts, args = getopt.getopt(argv,"he:sr:r:p:g",["experiment=","starting_run=", "runs=","population_size=","generations="])
    except getopt.GetoptError:
        print("run_experiment.py --experiment=<experiment name>(one of: SO, QN-MOEA, MNSLC) --starting_run=<starting_run> --runs=<runs> --population_size=<population size> --generations=<generations>")
        sys.exit()
    for opt, arg in opts:
        if opt == "-h":
            print("run_experiment.py --experiment=<experiment name>(one of: SO, QN-MOEA, MNSLC) --starting_run=<starting_run> --runs=<runs> --population_size=<population size> --generations=<generations>")
            sys.exit()
        if opt in ["-e", "--experiment"]:
            experiment = arg
            if experiment not in ["SO", "QN-MOEA", "MNSLC"]:
                print("run_experiment.py --experiment=<experiment name>(one of: SO, QN-MOEA, MNSLC) --starting_run=<starting_run> --runs=<runs> --population_size=<population size> --generations=<generations>")
                sys.exit()
        elif opt in ["-sr", "--starting_run"]:
            starting_run = int(arg)
        elif opt in ["-r", "--runs"]:
            runs = int(arg)
        elif opt in ["-p", "--population_size"]:
            pop_size = int(arg)
        elif opt in ["-g", "--generations"]:
            max_gens = int(arg)
    
    genotype_cls = BodyBrainGenotypeIndirect2
    phenotype_cls = SimplePhenotypeIndirect
    softbot_problem_cls = None
    # Creating an objectives dictionary
    objective_dict = ObjectiveDict()

    # Now specifying the objectives for the optimization.
    # Adding an objective named "fitness", which we want to maximize. This information is returned by Voxelyze
    # in a fitness .xml file, with a tag named "NormFinalDist"
    objective_dict.add_objective(name="fitness", maximize=True, tag="<FinalDist>")

    # Adding another objective called "num_voxels", which we want to minimize in order to minimize
    # the amount of material employed to build the robot, promoting at the same time non-trivial
    # morphologies.
    # This information can be computed in Python (it's not returned by Voxelyze, thus tag=None),
    # which is done by counting the non empty voxels (material != 0) composing the robot.
    objective_dict.add_objective(name="num_voxels", maximize=False, tag=None,
                                    node_func=np.count_nonzero, output_node_name="material")

    # This information is not returned by Voxelyze (tag=None): it is instead computed in Python.
    # We also specify how energy should be computed, which is done by counting the occurrences of
    # active materials (materials number 3 and 4)
    objective_dict.add_objective(name="active", maximize=False, tag=None,
                                    node_func=partial(count_occurrences, keys=[3, 4]),
                                    output_node_name="material")

    objective_dict.add_objective(name="passive", maximize=False, tag=None,
                            node_func=partial(count_occurrences, keys=[1, 2]),
                            output_node_name="material")
                            
    objective_dict.add_objective(name="unaligned_novelty", maximize=True, tag=None)

    if experiment == "SO":
        seeds_json = SEEDS_JSON_SO
        analytics_json = ANALYTICS_JSON_SO
        analytics_csv = ANALYTICS_JSON_SO.replace(".json", ".csv")
        run_dir = RUN_DIR_SO
        run_name = RUN_NAME_SO
        softbot_problem_cls = QualitySoftbotProblem
    
    elif experiment == "QN-MOEA":
        seeds_json = SEEDS_JSON_QN
        analytics_json = ANALYTICS_JSON_QN
        analytics_csv = ANALYTICS_JSON_QN.replace(".json", ".csv")
        run_dir = RUN_DIR_QN
        run_name = RUN_NAME_QN
        softbot_problem_cls = QualityNoveltySoftbotProblem

    elif experiment == "MNSLC":
        seeds_json = SEEDS_JSON_MNSLC
        analytics_json = ANALYTICS_JSON_MNSLC
        analytics_csv = ANALYTICS_JSON_MNSLC.replace(".json", ".csv")
        run_dir = RUN_DIR_MNSLC
        run_name = RUN_NAME_MNSLC
        softbot_problem_cls = MNSLCSoftbotProblem
        objective_dict.add_objective(name="fitnessX", maximize=True, tag="<finalDistX>")
        objective_dict.add_objective(name="fitnessY", maximize=True, tag="<finalDistY>")
        objective_dict.add_objective(name="aligned_novelty", maximize=True, tag=None)
        objective_dict.add_objective(name="unaligned_neighbors", maximize=True, tag=None)
        objective_dict.add_objective(name="nslc_quality", maximize=True, tag=None)
    
    runToSeedMapping = readFromJson(seeds_json)
    runsSoFar = countFileLines(analytics_json)
    if runsSoFar > 0:
        runToAnalyticsMapping = readFirstJson(analytics_json)
        firstRun = list(runToAnalyticsMapping.keys())
        firstRun = int(firstRun[0]) + runsSoFar
    else:
        firstRun = starting_run
    

    for run in range(firstRun - 1, runs):
        new_experiment = run + 1 == starting_run

        # Setting random seed
        print(f"Starting run: {run + 1}")
        if not str(run + 1) in runToSeedMapping.keys():
            runToSeedMapping[str(run + 1)] = uuid.uuid4().int & (1<<32)-1
            writeToJson(seeds_json, runToSeedMapping)
        random.seed(runToSeedMapping[str(run + 1)])  # Initializing the random number generator for reproducibility
        np.random.seed(runToSeedMapping[str(run + 1)])
        
        # Setting up the simulation object
        sim = Sim(dt_frac=DT_FRAC, simulation_time=SIM_TIME, fitness_eval_init_time=INIT_TIME)

        # Setting up the environment object
        env = Env(sticky_floor=0, time_between_traces=0)


        # Initializing a population of SoftBots
        my_pop = SoftbotPopulation(objective_dict, genotype_cls, phenotype_cls, pop_size=pop_size)
        pop = Population.new("X", my_pop)

        #Setting up Softbot optimization problem
        softbot_problem = softbot_problem_cls(sim, env, SAVE_POPULATION_EVERY, run_dir, run_name + str(run + 1), MAX_EVAL_TIME, TIME_TO_TRY_AGAIN, SAVE_LINEAGES, objective_dict)

        # Setting up our optimization
        algorithm = NSGA2(pop_size=pop_size, sampling=np.array(my_pop.individuals), mutation=SoftbotMutation(), crossover=DummySoftbotCrossover(), eliminate_duplicates=False)
        algorithm.setup(softbot_problem, termination=('n_gen', max_gens))
        analytics = QD_Analytics(run + 1, experiment)
        my_optimization = PopulationBasedOptimizerPyMOO(sim, env, algorithm, softbot_problem, analytics)

        # start optimization
        my_optimization.run(my_pop, max_hours_runtime=MAX_TIME, max_gens=max_gens, num_random_individuals=NUM_RANDOM_INDS, checkpoint_every=CHECKPOINT_EVERY, new_run = new_experiment)
        save_json(analytics_json, analytics.qd_history)
        df = analytics.to_dataframe()
        df.to_csv(analytics_csv, mode='a', header=not os.path.exists(analytics_csv), index = False)

    # runBodyBrain(runs, pop_size, max_gens, seeds_json, analytics_json, objective_dict, softbot_problem_cls, genotype_cls, phenotype_cls)
       



if __name__ == "__main__":
    main(sys.argv[1:])
