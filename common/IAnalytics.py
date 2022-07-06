
from abc import abstractmethod
from pymoo.core.callback import Callback

from evosoro_pymoo.common.IStart import IStarter


class IAnalytics(Callback, IStarter):
    @abstractmethod
    def start(self):
        pass

    def notify(self, algorithm, **kwargs):
        return super().notify(algorithm, **kwargs)