import abc


class CheckpointInterface(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'backup') and
                callable(subclass.backup))


@CheckpointInterface.register
class ICheckpoint:

    def backup(self, *args, **kwargs):
        """
            Execute any code for backing up the state of an object
        """
        pass
