
"""

    Based on the setup and code originally by:

    Cheney, N., MacCurdy, R., Clune, J., & Lipson, H. (2013).
    Unshackling evolution: evolving soft robots with multiple materials and a powerful generative encoding.
    In Proceedings of the 15th annual conference on Genetic and evolutionary computation (pp. 167-174). ACM.

    Related video: https://youtu.be/EXuR_soDnFo

"""

import glob
import re
import numpy
import os
import sys
import uuid
import random
import argparse
import logging
from functools import partial
from pymoo.algorithms.moo.nsga2 import NSGA2, RankAndCrowdingSurvival
from pymoo.core.population import Population
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.getenv('PYTHONPATH'))# Appending repo's root dir in the python path to enable subsequent imports

from common.Constants import *
from evosoro_pymoo.Algorithms.OptimizerPyMOO import PopulationBasedOptimizerPyMOO
from evosoro_pymoo.Algorithms.RankAndVectorFieldDiversitySurvival import RankAndVectorFieldDiversitySurvival
from evosoro_pymoo.Evaluators.PhysicsEvaluator import VoxcraftPhysicsEvaluator, VoxelyzePhysicsEvaluator
from evosoro_pymoo.Operators.Crossover import DummySoftbotCrossover
from evosoro_pymoo.Operators.Mutation import ME_SoftbotMutation, SoftbotMutation
from common.Utils import readFromJson, setRandomSeed, writeToJson, countFileLines, readFirstJson
from common.Analytics import QD_Analytics

from evosoro.base import Sim, Env, ObjectiveDict
from evosoro.tools.utils import count_occurrences
from evosoro.softbot import Population as SoftbotPopulation
from SoftbotProblemDefs import MNSLCSoftbotProblemGPU, QualitySoftbotProblem, QualityNoveltySoftbotProblem, MNSLCSoftbotProblem, NSLCSoftbotProblem, MESoftbotProblem
from Genotypes import BodyBrainGenotypeIndirect, SimplePhenotypeIndirect
from BodyBrainCommon import runBodyBrain



# create logger with __name__
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# create file handler which logs only warning level messages
fh = logging.FileHandler('experiments.log')
fh.setLevel(logging.WARNING)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)


def main(parser : argparse.ArgumentParser):


    argv = parser.parse_args()
    experiment = argv.experiment
    starting_run = argv.starting_run
    runs = argv.runs
    pop_size = argv.population_size
    max_gens = argv.generations
    physics_sim = argv.physics


    if experiment not in EXPERIMENT_TYPES or physics_sim not in PHYSICS_SIM_TYPES or starting_run <= 0 or starting_run > runs or pop_size <= 0 or runs <= 0 or pop_size <= 0 or max_gens <= 0:
        parser.print_help()
        sys.exit(2)
    
    
    genotype_cls = BodyBrainGenotypeIndirect
    phenotype_cls = SimplePhenotypeIndirect
    nsga2_survival = RankAndCrowdingSurvival()
    softbot_problem_cls = None
    physics_sim_cls = None

    # Creating an objectives dictionary
    objective_dict = ObjectiveDict()

    if physics_sim == 'CPU':

        # Now specifying the objectives for the optimization.
        # Adding an objective named "fitness". This information is returned by Voxelyze
        # in a fitness .xml file, with a tag named "NormFinalDist"
        objective_dict.add_objective(name="fitness", maximize=True, tag="<FinalDist>")

        # This information is not returned by Voxelyze (tag=None): it is instead computed in Python
        # Adding another objective called "num_voxels" for constraint reasons
        objective_dict.add_objective(name="num_voxels", maximize=True, tag=None,
                                        node_func=numpy.count_nonzero, output_node_name="material")

        
        objective_dict.add_objective(name="active", maximize=True, tag=None,
                                        node_func=partial(count_occurrences, keys=[3, 4]),
                                        output_node_name="material")

        objective_dict.add_objective(name="passive", maximize=True, tag=None,
                                node_func=partial(count_occurrences, keys=[1, 2]),
                                output_node_name="material")
                                
        objective_dict.add_objective(name="unaligned_novelty", maximize=True, tag=None)

        physics_sim_cls = VoxelyzePhysicsEvaluator

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

        elif experiment == "NSLC":
            seeds_json = SEEDS_JSON_NSLC
            analytics_json = ANALYTICS_JSON_NSLC
            analytics_csv = ANALYTICS_JSON_NSLC.replace(".json", ".csv")
            run_dir = RUN_DIR_NSLC
            run_name = RUN_NAME_NSLC
            softbot_problem_cls = MNSLCSoftbotProblem
            nsga2_survival = RankAndVectorFieldDiversitySurvival(orig_size_xyz=IND_SIZE)
            objective_dict.add_objective(name="unaligned_neighbors", maximize=True, tag=None)
            objective_dict.add_objective(name="nslc_quality", maximize=True, tag=None)

        elif experiment == "MNSLC":
            seeds_json = SEEDS_JSON_MNSLC
            analytics_json = ANALYTICS_JSON_MNSLC
            analytics_csv = ANALYTICS_JSON_MNSLC.replace(".json", ".csv")
            run_dir = RUN_DIR_MNSLC
            run_name = RUN_NAME_MNSLC
            softbot_problem_cls = NSLCSoftbotProblem
            nsga2_survival = RankAndVectorFieldDiversitySurvival(orig_size_xyz=IND_SIZE)
            objective_dict.add_objective(name="fitnessX", maximize=True, tag="<finalDistX>")
            objective_dict.add_objective(name="fitnessY", maximize=True, tag="<finalDistY>")
            objective_dict.add_objective(name="aligned_novelty", maximize=True, tag=None)
            objective_dict.add_objective(name="unaligned_neighbors", maximize=True, tag=None)
            objective_dict.add_objective(name="nslc_quality", maximize=True, tag=None)

    elif physics_sim == 'GPU':
        # Now specifying the objectives for the optimization.
        # Adding an objective named "fitness". This information is returned by Voxelyze
        # in a fitness .xml file, with a tag named "NormFinalDist"
        objective_dict.add_objective(name="fitness", maximize=True, tag="fitness_score")

        # This information is not returned by Voxelyze (tag=None): it is instead computed in Python
        # Adding another objective called "num_voxels" for constraint reasons
        objective_dict.add_objective(name="num_voxels", maximize=True, tag=None,
                                        node_func=numpy.count_nonzero, output_node_name="material")

        
        objective_dict.add_objective(name="active", maximize=True, tag=None,
                                        node_func=partial(count_occurrences, keys=[3, 4]),
                                        output_node_name="material")

        objective_dict.add_objective(name="passive", maximize=True, tag=None,
                                node_func=partial(count_occurrences, keys=[1, 2]),
                                output_node_name="material")
                                
        objective_dict.add_objective(name="unaligned_novelty", maximize=True, tag=None)
        objective_dict.add_objective(name="aligned_novelty", maximize=True, tag=None)

        objective_dict.add_objective(name="initialX", maximize=True, tag="initialCenterOfMass/x")
        objective_dict.add_objective(name="initialY", maximize=True, tag="initialCenterOfMass/y")
        objective_dict.add_objective(name="finalX", maximize=True, tag="currentCenterOfMass/x")
        objective_dict.add_objective(name="finalY", maximize=True, tag="currentCenterOfMass/y")
        objective_dict.add_objective(name="fitnessX", maximize=True, tag=None)
        objective_dict.add_objective(name="fitnessY", maximize=True, tag=None)
        
        objective_dict.add_objective(name="gene_diversity", maximize=True, tag=None)
        objective_dict.add_objective(name="control_gene_div", maximize=True, tag=None)
        objective_dict.add_objective(name="morpho_gene_div", maximize=True, tag=None)


        

        physics_sim_cls = VoxcraftPhysicsEvaluator

        if experiment == "SO":
            seeds_json = SEEDS_JSON_SO
            analytics_json = ANALYTICS_JSON_SO
            run_dir = RUN_DIR_SO
            run_name = RUN_NAME_SO
            softbot_problem_cls = QualitySoftbotProblem
        
        elif experiment == "QN-MOEA":
            seeds_json = SEEDS_JSON_QN
            analytics_json = ANALYTICS_JSON_QN
            run_dir = RUN_DIR_QN
            run_name = RUN_NAME_QN
            softbot_problem_cls = QualityNoveltySoftbotProblem

        elif experiment == "NSLC":
            seeds_json = SEEDS_JSON_NSLC
            analytics_json = ANALYTICS_JSON_NSLC
            run_dir = RUN_DIR_NSLC
            run_name = RUN_NAME_NSLC
            softbot_problem_cls = NSLCSoftbotProblem
            nsga2_survival = RankAndVectorFieldDiversitySurvival(orig_size_xyz=IND_SIZE)
            objective_dict.add_objective(name="unaligned_neighbors", maximize=True, tag=None)
            objective_dict.add_objective(name="nslc_quality", maximize=True, tag=None)

        elif experiment == "MAP-ELITES":
            seeds_json = SEEDS_JSON_ME
            analytics_json = ANALYTICS_JSON_ME
            run_dir = RUN_DIR_ME
            run_name = RUN_NAME_ME
            softbot_problem_cls = MESoftbotProblem

        elif experiment == "MNSLC":
            seeds_json = SEEDS_JSON_MNSLC
            analytics_json = ANALYTICS_JSON_MNSLC
            run_dir = RUN_DIR_MNSLC
            run_name = RUN_NAME_MNSLC
            softbot_problem_cls = MNSLCSoftbotProblemGPU
            nsga2_survival = RankAndVectorFieldDiversitySurvival(orig_size_xyz=IND_SIZE)
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

        # Setting random seed
        logger.info(f"Starting run: {run + 1}")
        if not str(run + 1) in runToSeedMapping.keys():
            runToSeedMapping[str(run + 1)] = uuid.uuid4().int & (1<<32)-1
            writeToJson(seeds_json, runToSeedMapping)
        
        # Initializing the random number generator for reproducibility
        numpy.random.seed(runToSeedMapping[str(run + 1)])
        random.seed(runToSeedMapping[str(run + 1)])  

        
        # Setting up the simulation object
        sim = Sim(dt_frac=DT_FRAC, simulation_time=SIM_TIME, fitness_eval_init_time=INIT_TIME)

        # Setting up the environment object
        env = Env(sticky_floor=0, time_between_traces=0, lattice_dimension=0.05)

        run_path = run_dir + str(run + 1)
        resume_run = False
        starting_gen = 1

        if os.path.exists(run_path) and os.path.isdir(run_path):
            response = input("****************************************************\n"
                            "** WARNING ** A directory named " + run_path + " may exist already.\n"
                            "Would you like to resume possibly pending run? (y/n): ")
            if not (("Y" in response) or ("y" in response)):
                print(f"Restarting run {run + 1}.\n"
                     "****************************************************\n\n")
            else:
                resume_run = True
                stored_bots = glob.glob(run_path + "/Gen_*")
                gen_lst = [int(str.lstrip(str(str.split(stored_bot, '_')[-1]), '0')) for stored_bot in stored_bots]
                gen_lst.sort()
                starting_gen = gen_lst[-1]
                print (f"Resuming run {run + 1} at generation {starting_gen}.\n"
                    "****************************************************\n")
        if starting_gen <= max_gens:
            start_success = False
            start_attempts = 0
            while not start_success and start_attempts < 5:
                start_attempts += 1
                # Setting up analytics
                analytics = QD_Analytics(run + 1, experiment, run_name, run_path, 'experiments', resume_run)
                analytics.set_generation(starting_gen)

                # Setting up physics simulation
                physics_sim = physics_sim_cls(sim, env, SAVE_POPULATION_EVERY, run_path, run_name, objective_dict, 
                                                max_gens, 0, max_eval_time= MAX_EVAL_TIME, time_to_try_again= TIME_TO_TRY_AGAIN, 
                                                save_lineages = SAVE_LINEAGES, resuming_run=resume_run)
                physics_sim.set_generation(starting_gen)

                # Setting up Softbot optimization problem
                softbot_problem = softbot_problem_cls(physics_sim, pop_size, run_path, orig_size_xyz=IND_SIZE) if not softbot_problem_cls is MESoftbotProblem else softbot_problem_cls(physics_sim, pop_size, run_path, resume_run, orig_size_xyz=IND_SIZE)

                # Initializing a population of SoftBots
                my_pop = SoftbotPopulation(objective_dict, genotype_cls, phenotype_cls, pop_size=pop_size)
                if resume_run:
                    resume_success = my_pop.start(run_path + "/pickledPops")
                    if not resume_success:
                        print("Insufficient data to resume run execution.\nRestarting run...")
                        resume_run = False
                        starting_gen = 1
                        continue
                else:
                    my_pop.start()
                Population.new("X", my_pop)
                start_success = True

            # Setting up optimization algorithm
            algorithm = NSGA2(pop_size=pop_size, sampling=numpy.array(my_pop.individuals), 
                            mutation=SoftbotMutation(my_pop.max_id) if experiment != "MAP-ELITES" else ME_SoftbotMutation(my_pop.max_id, pop_size), 
                            crossover=DummySoftbotCrossover(), survival=nsga2_survival, eliminate_duplicates=False)
            algorithm.setup(softbot_problem, termination=('n_gen', max_gens - starting_gen + 1))
            
            my_optimization = PopulationBasedOptimizerPyMOO(sim, env, algorithm, softbot_problem, analytics)
            my_optimization.start()

            # Start optimization
            my_optimization.run(my_pop)
            analytics.save_archives()
        

    # runBodyBrain(runs, pop_size, max_gens, seeds_json, analytics_json, objective_dict, softbot_problem_cls, genotype_cls, phenotype_cls)
    sys.exit()


if __name__ == "__main__":
    class CustomParser(argparse.ArgumentParser):
        def error(self, message):
            sys.stderr.write('error: %s\n' % message)
            self.print_help()
            sys.exit(2)
    parser = CustomParser()
    parser.add_argument('-e', '--experiment', type=str, default='SO', help="Experiment to run: SO(default), QN-MOEA, MNSLC")
    parser.add_argument('--starting_run', type=int, default=1, help="Run number to start from (use if existing data for experiment)")
    parser.add_argument('-r', '--runs', type=int, default=1, help="Number of runs of the experiment")
    parser.add_argument('-p', '--population_size', type=int, default=5, help="Size of the population")
    parser.add_argument('-g','--generations', type=int, default=20, help="Number of iterations the optimization algorithm will execute")
    parser.add_argument('--physics', type=str, default='CPU', help = "Type of physics engine to use: CPU (default), GPU")
    # parser.add_argument('-o', '--outputDir', type=str, default=None, help = "Path of the output log files")

    main(parser)
