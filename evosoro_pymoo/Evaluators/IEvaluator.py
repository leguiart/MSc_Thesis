import abc
from typing import Generic, List, TypeVar
from common.Utils import timeit

class EvaluatorInterface(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'evaluate') and 
                callable(subclass.evaluate))

T = TypeVar("T")

@EvaluatorInterface.register
class IEvaluator(Generic[T]):

    @timeit
    def evaluate(self, X : List[T], *args, **kwargs) -> List[T]:
        """Evaluates phenotypes of the elements of a list of individuals with a certain fitness metric
        X : list
            List of objects which contain a fitness metric and a phenotype
        """
        pass
