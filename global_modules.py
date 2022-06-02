
import time
import logging
from functools import wraps

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