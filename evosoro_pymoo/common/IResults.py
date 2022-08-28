import abc


class ResultsInterface(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'results') and
                callable(subclass.results))


@ResultsInterface.register
class IResults:

    def results(self, *args, **kwargs):
        """
            Execute any code for backing up the state of an object
        """
        pass