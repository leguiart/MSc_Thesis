import abc

class EvaluatorInterface(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'evaluate') and 
                callable(subclass.evaluate))


@EvaluatorInterface.register
class IEvaluator:

    def evaluate(self, X : list) -> list:
        """Evaluates phenotypes of the elements of a list of individuals with a certain fitness metric
        X : list
            List of objects which contain a fitness metric and a phenotype
        """
        pass
