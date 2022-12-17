
import numpy as np
from pymoo.core.survival import Survival

class MESurvival(Survival):

    def __init__(self) -> None:
        super().__init__(filter_infeasible=True)

    def _do(self, problem, pop, *args, n_survive=None, **kwargs):
        return pop[np.arange(0,n_survive)]