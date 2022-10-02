
import abc


class StateCleaningInterface(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'start') and 
                callable(subclass.start))


@StateCleaningInterface.register
class IStateCleaning:

    def clean(self, *args, **kwargs):
        """
            Execute internal state object cleaning action
        """
        pass
