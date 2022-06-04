import abc
from common.Utils import timeit

class EvaluatorInterface(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'evaluate') and 
                callable(subclass.evaluate))


@EvaluatorInterface.register
class IEvaluator:

    @timeit
    def evaluate(self, X : list, *args, **kwargs) -> list:
        """Evaluates phenotypes of the elements of a list of individuals with a certain fitness metric
        X : list
            List of objects which contain a fitness metric and a phenotype
        """
        pass
