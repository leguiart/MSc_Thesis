
import os
import sys

from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.getenv('PYTHONPATH'))# Appending repo's root dir in the python path to enable subsequent imports

from common.Utils import readFromJson, writeToJson

cache1 = readFromJson("experiments/physics_evaluator_cache.json")
cache2 = readFromJson("experiments/physics_evaluator_cache2.json")

cache1.update(cache2)

writeToJson("experiments/physics_evaluator_cache.json", cache1)
