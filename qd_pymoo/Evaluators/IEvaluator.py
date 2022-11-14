
import abc
from typing import Generic, List, TypeVar


class EvaluationFunctionInterface(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'evaluation_fn') and 
                callable(subclass.evaluation_fn))

T = TypeVar("T")

@EvaluationFunctionInterface.register
class IEvaluationFunction(Generic[T]):

    def evaluation_fn(self, X : List[T], *args, **kwargs) -> List[T]:
        """Evaluates phenotypes of the elements of a list of individuals with a certain fitness metric
        X : list
            List of objects which contain a fitness metric and a phenotype
        """
        pass
