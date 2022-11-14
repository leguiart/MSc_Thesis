
import glob
import os
import sys
import numpy as np
from dotenv import load_dotenv



load_dotenv()
sys.path.append(os.getenv('PYTHONPATH'))# Appending repo's root dir in the python path to enable subsequent imports

from experiments.Constants import *
from common.Utils import readFromPickle, writeToJson
from evosoro_pymoo.Evaluators.PhysicsEvaluator import VoxcraftPhysicsEvaluator, VoxelyzePhysicsEvaluator

physics_evaluator_cache = {}

for experiment_type in ["SO", "QN", "ME", "NSLC", "MNSLC"]:
    if experiment_type != "SO":
        run_dir_prefix = f"/media/leguiart/LuisExtra/ExperimentsData2/BodyBrain{experiment_type}Data"
    else:
        run_dir_prefix = "/media/leguiart/LuisExtra/ExperimentsData2/BodyBrainData"
    
    run_dirs = glob.glob(run_dir_prefix+"*")

    for run_dir in run_dirs:
        if os.path.isdir(run_dir):
            try:
                physics_evaluator = readFromPickle(f"{run_dir}/physics_evaluator_checkpoint.pickle")
            except:
                physics_evaluator = None
            if physics_evaluator:
                already_evaluated = physics_evaluator.already_evaluated
                fitness_index = 0
                objective_dict = physics_evaluator.objective_dict
                for indx, objective in objective_dict.items():
                    if objective['name'] == 'fitness':
                        fitness_index = indx
                        break
                
                for key in already_evaluated.keys():
                    if key in physics_evaluator_cache:
                        physics_evaluator_cache[key][0]+=[already_evaluated[key][fitness_index]]
                    else:
                        physics_evaluator_cache[key] = [[already_evaluated[key][fitness_index]], already_evaluated[key], fitness_index]
        
for key in physics_evaluator_cache.keys():
    fitness_array = np.array(physics_evaluator_cache[key][0])
    fitness_index = physics_evaluator_cache[key][2]
    objective_values = physics_evaluator_cache[key][1]
    if len(fitness_array) > 1:
        objective_values[fitness_index] = fitness_array.mean()
    else:
        objective_values[fitness_index] = fitness_array[0]
    if fitness_index != 0:
        print(fitness_index)
    physics_evaluator_cache[key] = objective_values
    

writeToJson('experiments/physics_evaluator_cache.json', physics_evaluator_cache)