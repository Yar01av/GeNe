from typing import List

import torch
from torch import no_grad, nn

from gene.optimisers.base import Optimiser
from gene.optimisers.division import DivisionOptimiser
from gene.selections.top_n import TopNSelection


class AnnealingOptimiser(Optimiser):
    def __init__(self,
                 target_func,
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
        :param target_func: A function that takes an outputs of the model and true values.
        :param selection: A selection instance that is used to select the best models.
        :param n_offsprings: How many offsprings does a model have.
        """

        self._std = init_std
        self._std_updater = std_updater
        self._div_optimiser = DivisionOptimiser(target_func=target_func,
                                                random_function=lambda shape: torch.normal(0, self._std, shape),
                                                selection=selection,
                                                n_offsprings=n_offsprings,
                                                keep_parents=keep_parents,
                                                device=device)

    def step(self, models: List[nn.Module], X, y_true) -> List[nn.Module]:
        new_models = self._div_optimiser.step(models, X, y_true)
        self._std = self._std_updater(self._std)

        return new_models
