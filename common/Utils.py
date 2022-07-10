
import random
import json
import os
import time
import logging
import numpy as np
import sys
from functools import wraps
from types import ModuleType, FunctionType
from gc import get_referents

from common.Constants import *


# create logger
module_logger = logging.getLogger(f"__main__.timeit")

def get_class_that_defined_method(method):
    method_name = method.__name__
    if method.__self__:    
        classes = [method.__self__.__class__]
    else:
        #unbound method
        classes = [method.im_class]
    while classes:
        c = classes.pop()
        if method_name in c.__dict__:
            return c
        else:
            classes = list(c.__bases__) + classes
    return None



def timeit(method):
    @wraps(method)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = method(*args, **kwargs)
        end_time = time.time()
        module_logger.info(f"{method.__qualname__} ({method.__module__}) => {(end_time-start_time)*1000} ms")

        return result

    return wrapper


def setRandomSeed(seed):
    random.seed(seed)  # Initializing the random number generator for reproducibility
    np.random.seed(seed)


def readFromJson(filename):
    if os.path.exists(filename):            
        with open(filename, 'r') as fp:           
            return json.load(fp)
    else:
        return {}


def writeToJson(filename, content):
    with open(filename, 'w') as fp:           
        json.dump(content, fp)



def save_json(filename, content):
    if os.path.exists(filename): 
        with open(filename, 'a') as fp:           
            fp.write('\n')
            json.dump(content, fp)
    else:
        with open(filename, 'w') as fp:           
            json.dump(content, fp)



def countFileLines(filename):
    if os.path.exists(filename): 
        with open(filename, 'r') as fh:
            count = 0
            for _ in fh:
                count += 1
        return count
    else:
        return 0



def readFirstJson(filename):
    if os.path.exists(filename): 
        with open(filename, 'r') as fh:
            line = fh.readline()
            j = json.loads(line)
            return j
    else:
        return {}



def maxFromList(l):
    max_elem = float('-inf')
    for elem in l:
        if elem > max_elem:
            max_elem = elem
    return max_elem




# Custom objects know their class.
# Function objects seem to know way too much, including modules.
# Exclude modules as well.
BLACKLIST = type, ModuleType, FunctionType


def getsize(obj):
    """sum size of object & members."""
    if isinstance(obj, BLACKLIST):
        raise TypeError('getsize() does not take argument of type: '+ str(type(obj)))
    seen_ids = set()
    size = 0
    objects = [obj]
    while objects:
        need_referents = []
        for obj in objects:
            if not isinstance(obj, BLACKLIST) and id(obj) not in seen_ids:
                seen_ids.add(id(obj))
                size += sys.getsizeof(obj)
                need_referents.append(obj)
        objects = get_referents(*need_referents)
    return size