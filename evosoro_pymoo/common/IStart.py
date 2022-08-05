
import abc


class StarterInterface(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'evaluate') and 
                callable(subclass.evaluate))


@StarterInterface.register
class IStarter:

    def start(self, **kwargs):
        """
            Execute any starter code needed to configure an object after creation
        """
        pass
