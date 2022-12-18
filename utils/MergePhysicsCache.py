

import os
import glob
import sys

# from dotenv import load_dotenv

# load_dotenv()
# sys.path.append(os.getenv('PYTHONPATH'))# Appending repo's root dir in the python path to enable subsequent imports

from utils import readFromJson, writeToJson

main_cache_path = '/media/leguiart/LuisExtra'


if os.path.exists(main_cache_path) and os.path.isdir(main_cache_path):
    stored_cache_paths = glob.glob(os.path.join(main_cache_path, "physics_evaluator_cache*.json"))
    cache_result = {}
    for cache_path in stored_cache_paths:
        if os.path.isfile(cache_path):
            print(cache_path)
            cache = readFromJson(cache_path)
            cache_result.update(cache)

writeToJson(os.path.join(main_cache_path, "merged_physics_cache.json"), cache_result)
