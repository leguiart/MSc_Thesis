
VOXELYZE_VERSION = '_voxcad'



NUM_RANDOM_INDS = 1  # Number of random individuals to insert each generation
MAX_GENS = 1000  # Number of generations
POPSIZE = 15  # Population size (number of individuals in the population)
IND_SIZE = (7, 7, 7)  # Bounding box dimensions (x,y,z). e.g. IND_SIZE = (6, 6, 6) -> workspace is a cube of 6x6x6 voxels
SIM_TIME = 5  # (seconds), including INIT_TIME!
INIT_TIME = 1
DT_FRAC = 1.  # Fraction of the optimal integration step. The lower, the more stable (and slower) the simulation.

TIME_TO_TRY_AGAIN = 30  # (seconds) wait this long before assuming simulation crashed and resending
MAX_EVAL_TIME = 60  # (seconds) wait this long before giving up on evaluating this individual
SAVE_LINEAGES = False
MAX_TIME = 8  # (hours) how long to wait before autosuspending
EXTRA_GENS = 0  # extra gens to run when continuing from checkpoint

CHECKPOINT_EVERY = 1  # How often to save an snapshot of the execution state to later resume the algorithm
SAVE_POPULATION_EVERY = 1  # How often (every x generations) we save a snapshot of the evolving population

#Simple GA experiment
RUN_DIR_SO = "BodyBrainData"  # Subdirectory where results are going to be generated
RUN_NAME_SO = "BodyBrain"
SEEDS_JSON_SO = RUN_DIR_SO + "_seeds.json"
ANALYTICS_JSON_SO = RUN_DIR_SO + "_analytics.json"
# ANALYTICS_FILENAME_SO = RUN_DIR_SO + "_analytics"

#QN-MOEA experiment
RUN_DIR_QN = "BodyBrainQNData"  # Subdirectory where results are going to be generated
RUN_NAME_QN = "BodyBrainQN"
SEEDS_JSON_QN = RUN_DIR_QN + "_seeds.json"
ANALYTICS_JSON_QN = RUN_DIR_QN + "_analytics.json"
# ANALYTICS_FILENAME_MNSLC = RUN_DIR_QN + "_analytics"


#M-NSLC experiment
RUN_DIR_MNSLC = "BodyBrainMNSLCData"  # Subdirectory where results are going to be generated
RUN_NAME_MNSLC = "BodyBrainMNSLC"
SEEDS_JSON_MNSLC = RUN_DIR_MNSLC + "_seeds.json"
ANALYTICS_JSON_MNSLC = RUN_DIR_MNSLC + "_analytics.json"
# ANALYTICS_FILENAME_MNSLC = RUN_DIR_MNSLC + "_analytics"
