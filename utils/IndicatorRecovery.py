
import glob
import os
import re
import sys
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Set, Tuple
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.getenv('PYTHONPATH'))# Appending repo's root dir in the python path to enable subsequent imports

from constants import *
from utils.utils import readFromPickle
from evosoro_pymoo.Evaluators.IEvaluator import IEvaluator
from experiments.Analytics.Analytics import QD_Analytics
from experiments.SoftbotProblemDefs import QualitySoftbotProblem


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# create file handler which logs only warning level messages
fh = logging.FileHandler('diversity.log')
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


class DummySurvival:
    def __init__(self):
        pass

class Algorithm:
    def __init__(self, problem):
        self.problem = problem
        self.survival = DummySurvival()
        self.is_initialized = True
        self.n_gen = 1


class SoftBot:
    def __init__(self):
        self.id = 0
        self.active = 0
        self.passive = 0
        self.finalX = 0.
        self.finalY = 0.
        self.finalZ = 0.
        self.initialX = 0.
        self.initialY = 0.
        self.initialZ = 0.
        self.genotype = None
        self.aligned_novelty = 0.
        self.unaligned_novelty = 0.
        self.fitness = 0.
        self.md5 = ""
        self.gene_diversity = 0.
        self.control_gene_div = 0.
        self.morpho_gene_div = 0.

class DummySoftBot:
    def __init__(self, x : SoftBot):
        self.X = x


class DummyPhysicsEvaluator(IEvaluator):
    def evaluate(self, X: List[SoftBot], *args, **kwargs) -> List[SoftBot]:
        return X

def df_to_population(df, ids_to_individuals):
    softbot_pop = []
    for _, row in df.iterrows():
        ind = SoftBot()
        ind.id = row['id']
        ind.md5 = row['md5']
        ind.fitness = row['fitness']
        ind.initialX = row['inipoint_x']
        ind.initialY = row['inipoint_y']
        ind.initialZ = row['inipoint_z']
        ind.finalX = row['endpoint_x']
        ind.finalY = row['endpoint_y']
        ind.finalZ = row['endpoint_z']
        ind.active = row['morphology_active']
        ind.passive = row['morphology_passive']
        ind.genotype = ids_to_individuals[ind.id]
        softbot_pop += [[DummySoftBot(ind)]]
    return softbot_pop

def main():
    base_experiments_path = "/media/leguiart/LuisExtra/ExperimentsData2"


    for experiment_type in ["SO", "QN", "ME", "NSLC", "MNSLC"]:

        if experiment_type != "SO":
            run_name = f"BodyBrain{experiment_type}"
        else:
            run_name = "BodyBrain"

        run_dir_prefix = os.path.join(base_experiments_path,f"{run_name}Data")
        run_dirs = glob.glob(run_dir_prefix+"*")
        run_dirs = list(filter(lambda dir : os.path.isdir(dir), run_dirs))
        
        if len(run_dirs) == 0:
            continue

        run_indices = {}
        for dir in run_dirs:
            indx = re.findall('[0-9]+', dir.split('/')[-1])
            if indx:
                run_indices[int(indx[0])] = dir

        original_indicators_csv_path = f"/media/leguiart/LuisExtra/ExperimentsData2/{run_name}.csv"
        indicators_csv = pd.read_csv(original_indicators_csv_path)
        grouped_by_run = [indicators_csv[indicators_csv['run'] == i] for i in run_indices.keys()]

        for j, item in enumerate(run_indices.items()):
            run_index, run_dir = item
            run_df = grouped_by_run[j]

            analytics = QD_Analytics(run_index, experiment_type, f"{run_name}_recovered_", run_dir, base_experiments_path)
            analytics.start(resuming_run = False, isNewExperiment = False)
            physics_sim = DummyPhysicsEvaluator()
            softbot_problem = QualitySoftbotProblem(physics_sim, 30, run_dir, orig_size_xyz=IND_SIZE)
            softbot_problem.start(resuming_run = False)
            algorithm = Algorithm(softbot_problem)

            stored_bots = glob.glob(run_dir + "/Gen_*")
            gen_lst = [int(str.lstrip(str(str.split(stored_bot, '_')[-1]), '0')) for stored_bot in stored_bots]
            gen_lst.sort()
            max_gens = len(gen_lst)
            run_not_included = False 
            softbot_pop = None

            for gen in gen_lst:
                gen_path = os.path.join(run_dir, f"Gen_{gen:04d}")
                if os.path.isdir(gen_path):
                    nn_backup_path = os.path.join(gen_path, f"Gen_{gen:04d}_networks.pickle")
                    
                    try:
                        generation_nns = readFromPickle(nn_backup_path)
                    except:
                        generation_nns = None

                    if generation_nns:
                        algorithm.n_gen = gen
                        ids_to_individuals = {individual[0] : individual[1] for individual in generation_nns}
                        d = {"id" : list(ids_to_individuals.keys())}
                        df = pd.DataFrame(d)
                        try:
                            df_joined = df.join(run_df.set_index('id'), on='id', lsuffix='_caller', rsuffix='_other')
                            softbot_pop_mat = df_to_population(df_joined, ids_to_individuals)
                            softbot_problem._evaluate(softbot_pop_mat, {"F":[], "G":[]})
                            softbot_pop = [vec[0] for vec in softbot_pop_mat]
                            lst_pop = [ind.X for ind in softbot_pop]
                            softbot_problem.clean(lst_pop, pop_size = len(lst_pop)//2)
                            analytics.notify(algorithm, pop = softbot_pop[:len(softbot_pop)//2], child_pop = softbot_pop[len(softbot_pop)//2:])  
                        except:
                            logger.warning(f"Run {run_index} of {experiment_type} experiment, needs to be repeated")
                            run_not_included = True
                            break                   
                    else:
                        if gen == max_gens:
                            res_set_path = os.path.join(run_dir, "results_set.pickle")

                            try:
                                res_set = readFromPickle(res_set_path)
                            except:
                                res_set = None

                            if res_set:
                                algorithm.n_gen = gen
                                res_pop_mat = [[individual] for individual in res_set['res'].pop]
                                softbot_problem._evaluate(res_pop_mat, {"F":[], "G":[]})
                                res_pop = [vec[0] for vec in res_pop_mat]
                                analytics.notify(algorithm, pop = res_pop, child_pop = softbot_pop[len(softbot_pop)//2:])
                            
                            # softbot_problem.backup(pickle_nov_archive = True)
                            analytics.save_archives(algorithm)  
                        else:
                            run_not_included = True
                            logger.warning(f"Run {run_index} of {experiment_type} experiment, needs to be repeated")
                        break

            if run_not_included:
                continue
 
                


if __name__ == "__main__":
    main()
