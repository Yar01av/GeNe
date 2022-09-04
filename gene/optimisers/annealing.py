from typing import List, Callable

import torch
from torch import no_grad, nn

from gene.optimisers.base import Optimiser
from gene.optimisers.division import DivisionOptimiser
from gene.selections.top_n import TopNSelection


class AnnealingOptimiser(Optimiser):
    def __init__(self,
                 init_std,
                 std_updater,
                 selection=TopNSelection(10),
                 n_offsprings=2,
                 keep_parents=True,
                 device="cpu"):
        """
        Optimiser that relies on simulated annealing to improve the weights of the networks.

        :param init_std: The initial standard deviation of the mutations.
        :param std_updater: A function that takes the current standard deviation and returns a new one. This is done
        after a new generation of models is produced.
        :param selection: A selection instance that is used to select the best models.
        :param n_offsprings: How many offsprings does a model have.
        """

        self._std = init_std
        self._std_updater = std_updater
        self._div_optimiser = DivisionOptimiser(random_function=lambda shape: torch.normal(0, self._std, shape),
                                                selection=selection,
                                                n_offsprings=n_offsprings,
                                                keep_parents=keep_parents,
                                                device=device)

    def step(self, models: List[nn.Module], loss_function: Callable[[nn.Module], float]) -> List[nn.Module]:
        new_models = self._div_optimiser.step(models, loss_function)
        self._std = self._std_updater(self._std)

        return new_models
