
import abc


class StarterInterface(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'start') and 
                callable(subclass.start))


@StarterInterface.register
class IStarter:

    def start(self, **kwargs):
        """
            Execute any starter code needed to configure an object after creation
        """
        pass
