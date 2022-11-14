
"""

    Based on the setup and code originally by:

    Cheney, N., MacCurdy, R., Clune, J., & Lipson, H. (2013).
    Unshackling evolution: evolving soft robots with multiple materials and a powerful generative encoding.
    In Proceedings of the 15th annual conference on Genetic and evolutionary computation (pp. 167-174). ACM.

    Related video: https://youtu.be/EXuR_soDnFo

"""

import os
import sys
import glob
import uuid
import logging
import argparse
import subprocess
import numpy as np
from functools import partial
from pymoo.core.sampling import Sampling
from pymoo.operators.selection.tournament import TournamentSelection
from pymoo.algorithms.moo.nsga2 import NSGA2, RankAndCrowdingSurvival, binary_tournament
from pymoo.algorithms.soo.nonconvex.ga import GA, FitnessSurvival, comp_by_cv_and_fitness

# from dotenv import load_dotenv
# load_dotenv()
# sys.path.append(os.getenv('PYTHONPATH'))# Appending repo's root dir in the python path to enable subsequent imports

from Constants import *
from evosoro_pymoo.Evaluators.GenotypeDiversityEvaluator import GenotypeDiversityEvaluator
from qd_pymoo.Evaluators.NoveltyEvaluator import NSLCEvaluator, NoveltyEvaluatorKD
from qd_pymoo.Algorithm.Optimizer import PopulationBasedOptimizer
from qd_pymoo.Algorithm.ME_Archive import MAP_ElitesArchive
from qd_pymoo.Algorithm.ME_Selection import MESelection
from qd_pymoo.Algorithm.ME_Survival import MESurvival

from evosoro_pymoo.Algorithms.RankAndVectorFieldDiversitySurvival import RankAndVectorFieldDiversitySurvival
from evosoro_pymoo.Evaluators.PhysicsEvaluator import BaseSoftBotPhysicsEvaluator, VoxcraftPhysicsEvaluator, VoxelyzePhysicsEvaluator
from evosoro_pymoo.Operators.Crossover import DummySoftbotCrossover
from evosoro_pymoo.Operators.Mutation import ME_SoftbotMutation, SoftbotMutation
from common.Utils import readFromJson,  saveToPickle, writeToJson
from analytics.Analytics import QD_Analytics

from evosoro.base import Sim, Env, ObjectiveDict
from evosoro.tools.utils import count_occurrences
from evosoro.softbot import Population as SoftbotPopulation, SoftBot
from SoftbotProblemDefs import SoftBotProblemFitness, SoftBotProblemFitnessNovelty, SoftBotProblemME, SoftBotProblemNSLC
from Genotypes import BodyBrainGenotypeIndirect, SimplePhenotypeIndirect


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

def extract_morpho(x : SoftBot):
    return [x.active, x.passive]  

def aligned_vector(a : SoftBot):
    return np.array([a.finalX - a.initialX, a.finalY - a.initialY, a.finalZ - a.initialZ])

def preconfig_objects(experiment : str, physics_sim : BaseSoftBotPhysicsEvaluator, unaligned_novelty_evaluator : NoveltyEvaluatorKD, 
                      me_evaluator : MAP_ElitesArchive, genotypeDiversityEvaluator : GenotypeDiversityEvaluator):
    ga_survival = RankAndCrowdingSurvival()
    ga_selection = TournamentSelection(func_comp=binary_tournament)
    if experiment == "SO":
        ga_survival = FitnessSurvival()
        ga_selection = TournamentSelection(func_comp=comp_by_cv_and_fitness)
        softbot_problem = SoftBotProblemFitness(physics_evaluator=physics_sim)
    elif experiment == "QN-MOEA":
        softbot_problem = SoftBotProblemFitnessNovelty(physics_evaluator=physics_sim, novelty_archive=unaligned_novelty_evaluator)
    elif experiment == "NSLC":
        unaligned_novelty_evaluator = NSLCEvaluator("unaligned_nslc_softbot", novelty_threshold=12., k_neighbors=300, 
                                          novelty_floor=1., max_novelty_archive_size=1500, 
                                          vector_extractor=extract_morpho)
        ga_survival = RankAndVectorFieldDiversitySurvival(genotypeDiversityEvaluator=genotypeDiversityEvaluator)
        softbot_problem = SoftBotProblemNSLC(physics_evaluator=physics_sim, nslc_archive=unaligned_novelty_evaluator)

    elif experiment == "MAP-ELITES":
        ga_selection = MESelection(me_archive=me_evaluator)
        ga_survival = MESurvival()
        softbot_problem = SoftBotProblemME(physics_evaluator=physics_sim, me_archive=me_evaluator)

    return softbot_problem, ga_survival, ga_selection


class PopulationSampler(Sampling):
    def __init__(self, objective_dict, genotype_cls, phenotype_cls, pop_size) -> None:
        self.my_pop = SoftbotPopulation(objective_dict, genotype_cls, phenotype_cls, pop_size=pop_size)
        super().__init__()

    def _do(self, problem, n_samples, **kwargs):
        # Initializing a population of SoftBots
        self.my_pop.start()
        my_pop_vec = np.array(self.my_pop.individuals)
        return my_pop_vec.reshape((my_pop_vec.shape[0], 1))
        

def main(parser : argparse.ArgumentParser):
    argv = parser.parse_args()
    experiment = argv.experiment
    starting_run = argv.starting_run
    runs = argv.runs
    pop_size = argv.population_size
    max_gens = argv.generations
    physics_sim = argv.physics
    usePhysicsCache = argv.physics_cache
    save_checkpoint = argv.save_checkpoint
    save_every = argv.save_every
    save_networks = argv.save_population
    skip_existing = argv.skip_existing

    if experiment not in EXPERIMENT_TYPES or physics_sim not in PHYSICS_SIM_TYPES or starting_run <= 0 or starting_run > runs or pop_size <= 0 or runs <= 0 or pop_size <= 0 or max_gens <= 0:
        parser.print_help()
        sys.exit(2)
    
    genotype_cls = BodyBrainGenotypeIndirect
    phenotype_cls = SimplePhenotypeIndirect
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
                                        node_func=np.count_nonzero, output_node_name="material")

        
        objective_dict.add_objective(name="active", maximize=True, tag=None,
                                        node_func=partial(count_occurrences, keys=[3, 4]),
                                        output_node_name="material")

        objective_dict.add_objective(name="passive", maximize=True, tag=None,
                                node_func=partial(count_occurrences, keys=[1, 2]),
                                output_node_name="material")
                                
        objective_dict.add_objective(name="unaligned_novelty", maximize=True, tag=None)
        objective_dict.add_objective(name="aligned_novelty", maximize=True, tag=None)

        objective_dict.add_objective(name="initialX", maximize=True, tag="<initialCenterOfMassX>")
        objective_dict.add_objective(name="initialY", maximize=True, tag="<initialCenterOfMassY>")
        objective_dict.add_objective(name="finalX", maximize=True, tag="<currentCenterOfMassX>")
        objective_dict.add_objective(name="finalY", maximize=True, tag="<currentCenterOfMassY>")
        objective_dict.add_objective(name="fitnessX", maximize=True, tag=None)
        objective_dict.add_objective(name="fitnessY", maximize=True, tag=None)
        
        objective_dict.add_objective(name="gene_diversity", maximize=True, tag=None)
        objective_dict.add_objective(name="control_gene_div", maximize=True, tag=None)
        objective_dict.add_objective(name="morpho_gene_div", maximize=True, tag=None)

        physics_sim_cls = VoxelyzePhysicsEvaluator
    elif physics_sim == 'GPU':
        # Now specifying the objectives for the optimization.
        # Adding an objective named "fitness". This information is returned by Voxelyze
        # in a fitness .xml file, with a tag named "fitness_score"
        objective_dict.add_objective(name="fitness", maximize=True, tag="fitness_score")
        # This information is not returned by voxcraft (tag=None): it is instead computed in Python
        objective_dict.add_objective(name="num_voxels", maximize=True, tag=None,
                                        node_func=np.count_nonzero, output_node_name="material")       
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
        objective_dict.add_objective(name="initialZ", maximize=True, tag="initialCenterOfMass/z")
        objective_dict.add_objective(name="finalX", maximize=True, tag="currentCenterOfMass/x")
        objective_dict.add_objective(name="finalY", maximize=True, tag="currentCenterOfMass/y")
        objective_dict.add_objective(name="finalZ", maximize=True, tag="currentCenterOfMass/z")
        objective_dict.add_objective(name="fitnessX", maximize=True, tag=None)
        objective_dict.add_objective(name="fitnessY", maximize=True, tag=None)
        objective_dict.add_objective(name="gene_diversity", maximize=True, tag=None)
        objective_dict.add_objective(name="control_gene_div", maximize=True, tag=None)
        objective_dict.add_objective(name="morpho_gene_div", maximize=True, tag=None)

        physics_sim_cls = VoxcraftPhysicsEvaluator

    if experiment == "SO":
        seeds_json = SEEDS_JSON_SO
        run_dir = RUN_DIR_SO
        run_name = RUN_NAME_SO
    
    elif experiment == "QN-MOEA":
        seeds_json = SEEDS_JSON_QN
        run_dir = RUN_DIR_QN
        run_name = RUN_NAME_QN

    elif experiment == "NSLC":
        seeds_json = SEEDS_JSON_NSLC
        run_dir = RUN_DIR_NSLC
        run_name = RUN_NAME_NSLC

    elif experiment == "MAP-ELITES":
        seeds_json = SEEDS_JSON_ME
        run_dir = RUN_DIR_ME
        run_name = RUN_NAME_ME


    runToSeedMapping = readFromJson(seeds_json)
    firstRun = starting_run
    
    for run in range(firstRun - 1, runs):
        
        # Setting random seed
        logger.info(f"Starting run: {run + 1}")
        if not str(run + 1) in runToSeedMapping.keys():
            runToSeedMapping[str(run + 1)] = uuid.uuid4().int & (1<<32)-1
            writeToJson(seeds_json, runToSeedMapping)

        random_seed = runToSeedMapping[str(run + 1)]

        # Setting up the simulation object
        sim = Sim(dt_frac=DT_FRAC, simulation_time=SIM_TIME, fitness_eval_init_time=INIT_TIME)

        # Setting up the environment object
        env = [Env(sticky_floor=0, time_between_traces=0, lattice_dimension=0.015)]

        run_path = run_dir + str(run + 1)

        resume_run = False
        starting_gen = 1

        if os.path.exists(run_path) and os.path.isdir(run_path):
            if skip_existing:
                continue
            response = input("****************************************************\n"
                            "** WARNING ** A directory named " + run_path + " may exist already.\n"
                            "Would you like to resume possibly pending run? (y/n): ")
            if not (("Y" in response) or ("y" in response)):
                response = input("****************************************************\n"
                "Would you like to skip the run? (y/n): ")
                if not (("Y" in response) or ("y" in response)):
                    print(f"Restarting run {run + 1}.\n"
                     "****************************************************\n\n")
                else:
                    print(f"Skipping run {run + 1}.\n"
                     "****************************************************\n\n")
                    continue
            else:
                resume_run = True
                stored_bots = glob.glob(run_path + "/Gen_*")
                gen_lst = [int(str.lstrip(str(str.split(stored_bot, '_')[-1]), '0')) for stored_bot in stored_bots]
                gen_lst.sort()
                starting_gen = gen_lst[-1]
                print (f"Resuming run {run + 1} at generation {starting_gen}.\n"
                    "****************************************************\n")
        
        if starting_gen <= max_gens:
            unaligned_novelty_evaluator = NoveltyEvaluatorKD("unaligned_novelty_softbot", novelty_threshold=12., 
                                    k_neighbors=300, novelty_floor=1., max_novelty_archive_size=1500, 
                                    vector_extractor=extract_morpho)
            aligned_novelty_evaluator = NoveltyEvaluatorKD("aligned_novelty_softbot", novelty_threshold=12., 
                                    k_neighbors=300, novelty_floor=1., max_novelty_archive_size=1500, 
                                    vector_extractor=aligned_vector)
            genotypeDiversityEvaluator = GenotypeDiversityEvaluator(orig_size_xyz=IND_SIZE)
            me_evaluator = MAP_ElitesArchive("me_softbot", np.array([0.,0.]), np.array([125., 125.]), np.array([25, 25]), extract_descriptors_func=extract_morpho)

            # Setting up analytics
            analytics = QD_Analytics(run + 1, experiment, run_name, run_path, 'experiments')

            # Setting up physics simulation
            physics_sim = physics_sim_cls(sim, env, SAVE_POPULATION_EVERY, run_path, run_name, objective_dict, 
                                            max_gens, 0, max_eval_time= MAX_EVAL_TIME, time_to_try_again= TIME_TO_TRY_AGAIN, 
                                            save_lineages = SAVE_LINEAGES)
            physics_sim.start(usePhysicsCache = usePhysicsCache, resuming_run = resume_run)
            # Setting up Softbot optimization problem
            softbot_problem, ga_survival, ga_selection = preconfig_objects(experiment, physics_sim, unaligned_novelty_evaluator, me_evaluator, genotypeDiversityEvaluator)

            # Setting up optimization algorithm
            softbot_mutation = SoftbotMutation(pop_size, objective_dict)
            if experiment in ["SO", "MAP-ELITES"]:
                algorithm = GA(pop_size=pop_size,
                                mutation=softbot_mutation, 
                                crossover=DummySoftbotCrossover(), 
                                survival=ga_survival, 
                                selection=ga_selection, 
                                eliminate_duplicates=False,
                                sampling=PopulationSampler(objective_dict, genotype_cls, phenotype_cls, pop_size))
            else:
                algorithm = NSGA2(pop_size=pop_size,
                                mutation=softbot_mutation, 
                                crossover=DummySoftbotCrossover(), 
                                survival=ga_survival, 
                                selection=ga_selection, 
                                eliminate_duplicates=False,
                                sampling=PopulationSampler(objective_dict, genotype_cls, phenotype_cls, pop_size))
            algorithm.setup(softbot_problem, termination=('n_gen', max_gens + 1), seed=random_seed)
            my_optimization = PopulationBasedOptimizer(algorithm, softbot_problem)

            # Start optimization
            results = my_optimization.run()
            # Save returned results to pickle
            saveToPickle(os.path.join(run_path, "results_set.pickle"), results)
            # Save physics sim backup regardless of checkpoints being activated or not,
            # in case of recovery of physics sim cache being done later on.
            physics_sim.backup()
            physics_evaluator_cache = physics_sim.already_evaluated
            writeToJson('physics_evaluator_cache.json', physics_evaluator_cache)

            # analytics.save_archives(algorithm)
        
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
    parser.add_argument('--physics_cache', action='store_true', help = "Use existing physics cache")
    parser.add_argument('--save_checkpoint', action='store_true', help = "Use to save checkpoints from which to continue in case the program is stopped")
    parser.add_argument('-se', '--save_every', type=int, default=1, help="Save checkpoints every given number of generations")
    parser.add_argument('--save_population', action='store_true', help = "Use to save population each generation")
    parser.add_argument('--skip_existing', action='store_true', help = "Use to skip any run with stored data in existance")

    main(parser)
