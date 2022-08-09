
from abc import abstractmethod
from pymoo.core.callback import Callback

from evosoro_pymoo.common.ICheckpoint import ICheckpoint
from evosoro_pymoo.common.IRecoverFromFile import IFileRecovery
from evosoro_pymoo.common.IStart import IStarter


class IAnalytics(Callback, ICheckpoint, IStarter, IFileRecovery):
    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def file_recovery(self, *args, **kwargs):
        pass

    @abstractmethod
    def backup(self, args, **kwargs):
        pass

    def notify(self, algorithm, **kwargs):
        return super().notify(algorithm, **kwargs)