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
            Fetch relevant data for result gathering
        """
        pass