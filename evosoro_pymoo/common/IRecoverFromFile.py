import abc


class RecoverFromFileInterface(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'file_recovery') and
                callable(subclass.file_recovery))


@RecoverFromFileInterface.register
class IFileRecovery:

    def file_recovery(self, *args, **kwargs):
        """
            Recover object from a file/files
        """
        pass